"""Rank the 35 query notices for the frozen-base and fully trained readout.

The 668-candidate development panel is the existing full9 similar-program task.
Each candidate pair gets one official reference-backbone forward; the original
and trained FP32 readouts score the same marker features. No historical BF16
scores are reused as a matched base.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file

HERE = Path(__file__).resolve().parent
MODEL = Path(r"D:/dev_eval_team/_models_harness/pplx-decider-v1.2")
MODEL_SRC = MODEL / "source" / "src"
PYTHON = Path(r"D:/dev_eval_team/_decider_env_harness/Scripts/python.exe")
DATA = HERE.parent / "decider_identity_v1"
INPUT = Path(r"D:/dev_eval_team/jh_dev/kim_program_inventory_v1/runs/044/pool668_newreview_harness/texts.json")
HISTORICAL = Path(r"D:/dev_eval_team/jh_dev/kim_program_inventory_v1/runs/044/decision_model_harness/decider_eval_pool668_27b_full9_harness.json")
INPUT_SHA256 = "649a57f7dd5ed6bd03d4cfe0305ea652458d9fcd94d5150b79eaf03628f5fba4"
QUESTION = {
    "type": "noul",
    "instructions": "공고 A와 공고 B는 지원 목적과 지원 대상이 비슷한 유사 사업인가?",
    "criteria": {"true": "유사한 사업이다", "false": "유사한 사업이 아니다"},
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def load_panel() -> tuple[dict, list[str], list[str], list[int]]:
    if sha(INPUT) != INPUT_SHA256:
        raise ValueError("Frozen 668 full9 text hash mismatch")
    data = json.loads(INPUT.read_text(encoding="utf-8"))
    keys = data["keys"]
    texts = [data["texts"]["chunk"][key] for key in keys]
    if len(keys) != 668 or len(set(keys)) != 668 or not all(texts):
        raise ValueError("Unexpected candidate pool")
    index = {key: i for i, key in enumerate(keys)}
    query_indices = sorted({index[k] for pair in data["eval_groups"].values() for k in pair})
    if len(query_indices) != 35 or len(data["eval_groups"]) != 30:
        raise ValueError("Expected 35 query notices and 30 positive group pairs")
    return data, keys, texts, query_indices


def request(a: int, b: int, texts: list[str]) -> dict:
    return {
        "state": json.dumps({"공고 A": texts[a], "공고 B": texts[b]}, ensure_ascii=False),
        "questions": {"similar": QUESTION},
    }


def summarize(data: dict, keys: list[str], matrix: np.ndarray) -> dict:
    index = {key: i for i, key in enumerate(keys)}
    query_indices = {index[k] for pair in data["eval_groups"].values() for k in pair}
    for a in query_indices:
        if not np.isnan(matrix[a, a]) or not np.isfinite(np.delete(matrix[a], a)).all():
            raise ValueError(f"Missing or invalid row scores for query {keys[a]}")

    def rank(a: int, b: int) -> float:
        row = np.delete(matrix[a], a)
        value = matrix[a, b]
        return float(1 + np.sum(row > value) + (np.sum(row == value) - 1) / 2)

    pairs = []
    groups: dict[str, set[str]] = {}
    for name, (x, y) in data["eval_groups"].items():
        a, b = index[x], index[y]
        pairs.append({"pair": name, "group": name.split("#")[0], "a": x, "b": y,
                      "rank_b_from_a": rank(a, b), "rank_a_from_b": rank(b, a),
                      "score_ab": float(matrix[a, b]), "score_ba": float(matrix[b, a])})
        groups.setdefault(name.split("#")[0], set()).update((x, y))
    directed = [p[k] for p in pairs for k in ("rank_b_from_a", "rank_a_from_b")]
    last = {x: max(rank(index[x], index[y]) for y in members if y != x)
            for members in groups.values() for x in members}
    return {
        "summary": {
            "n_pairs": len(pairs), "n_directed": len(directed), "n_groups": len(groups),
            "n_query_docs": len(last), "top5_share": float(np.mean(np.asarray(directed) <= 5)),
            "top30_share": float(np.mean(np.asarray(directed) <= 30)),
            "directed_rank_mean": float(np.mean(directed)),
            "directed_rank_median": float(np.median(directed)),
            "directed_rank_max": float(max(directed)),
            "last_correct_rank_mean": float(np.mean(list(last.values()))),
            "last_correct_rank_median": float(np.median(list(last.values()))),
            "last_correct_rank_max": float(max(last.values())),
        },
        "pairs": pairs,
        "last_correct_rank": last,
    }


def official_marker_logits(decider, chunks: list, base_weight: torch.Tensor,
                           trained_weight: torch.Tensor, left_pad) -> tuple[list, list]:
    """Apply both FP32 heads to marker states gathered with the official reference path."""
    device = decider.device
    inputs = {key: value.to(device) for key, value in left_pad(chunks, decider.encoder.pad_id).items()}
    hidden = decider.backbone(**inputs, use_cache=False).last_hidden_state
    offsets = [inputs["input_ids"].shape[1] - chunk.length for chunk in chunks]
    rows = torch.cat([torch.full((len(chunk.keys),), i, dtype=torch.long) for i, chunk in enumerate(chunks)]).to(device)
    cols = torch.cat([chunk.markers + offsets[i] for i, chunk in enumerate(chunks)]).to(device)
    marker_hidden = hidden[rows, cols].float()
    base_logits = F.linear(marker_hidden, base_weight[:2].float())
    full_logits = F.linear(marker_hidden, trained_weight[:2].float())
    return base_logits.float().cpu().tolist(), full_logits.float().cpu().tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-dir", type=Path, required=True)
    parser.add_argument("--full-status", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-tokens", type=int, default=4096)
    parser.add_argument("--max-length", type=int, default=2048)
    args = parser.parse_args()
    if args.batch_tokens < 1 or args.max_length < 1:
        parser.error("Token limits must be positive")
    full_dir = args.full_dir.resolve()
    head_eval_path = full_dir / "head_evaluation.json"
    after_path = full_dir / "readout_after.safetensors"
    status_path = args.full_status.resolve()
    if not head_eval_path.is_file() or not after_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("Full head result/status is not ready")
    full_status = json.loads(status_path.read_text(encoding="utf-8"))
    head_eval = json.loads(head_eval_path.read_text(encoding="utf-8"))
    if full_status.get("status") != "complete":
        raise RuntimeError(f"Full training status is not complete: {full_status.get('status')}")
    if not (head_eval.get("full_coverage") is True and head_eval.get("head_only") is True
            and head_eval.get("completed_epochs", 0) >= 1
            and head_eval.get("train_rows_unique_per_epoch") == head_eval.get("train_rows_per_epoch")):
        raise RuntimeError("The full head evaluation does not prove one complete unique-data pass")
    if sha(INPUT) != INPUT_SHA256:
        raise ValueError("External panel hash mismatch")
    training_manifest = json.loads((DATA / "manifest.json").read_text(encoding="utf-8"))
    if head_eval.get("train_rows_per_epoch") != training_manifest["pairs"]["train"]:
        raise ValueError("The trained head did not cover the current prepared train set")
    expected_base_readout = sha(MODEL / "readout.safetensors")
    if head_eval.get("config", {}).get("base_readout_sha256") != expected_base_readout:
        raise ValueError("Full head was initialized from a different base readout")
    data, keys, texts, queries = load_panel()
    manifest = {
        "input_sha256": INPUT_SHA256, "question": QUESTION, "input_mode": "full9",
        "queries": len(queries), "candidates": len(keys), "positive_pairs": len(data["eval_groups"]),
        "score": "FP32 logit(true)-logit(false), no temperature",
        "max_length": args.max_length, "batch_tokens": args.batch_tokens,
        "full_status_sha256": sha(status_path), "head_evaluation_sha256": sha(head_eval_path),
        "readout_after_sha256": sha(after_path), "base_readout_sha256": expected_base_readout,
        "model_manifest_sha256": sha(MODEL / "decision_config.json"),
        "reference": "official Decider Encoder, left pad, reference Qwen3_5Model attention masks; both readouts use same marker hidden states",
        "historical_base_reused": False,
        "code_sha256": sha(Path(__file__).resolve()),
    }
    out = args.output_dir.resolve()
    sys.path.insert(0, str(HERE.parent / "decider_section_v2"))
    from run_experiment import gpu_free
    try:
        gpu_free()
    except RuntimeError:
        # Restricted WDDM sessions can redact every process name, including the desktop.
        import subprocess
        names = subprocess.check_output(["nvidia-smi","--query-compute-apps=process_name","--format=csv,noheader"],text=True).splitlines()
        if not names or any(name.strip() != "[Insufficient Permissions]" for name in names):
            raise
        for _ in range(3):
            used, util = map(int, subprocess.check_output(["nvidia-smi","-i","0","--query-gpu=memory.used,utilization.gpu","--format=csv,noheader,nounits"],text=True).strip().split(","))
            if used > 1024 or util > 30:
                raise
            time.sleep(1)
    out.mkdir(parents=True, exist_ok=False)
    atomic_json(out / "manifest.json", manifest)
    atomic_json(out / "status.json", {"status": "loading_model", "manifest": manifest})
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, str(MODEL_SRC))
    from pplx_decider import Decider
    from pplx_decider.protocol import left_pad

    from unittest.mock import patch
    from safetensors import safe_open
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model
    def load_cuda_checkpoint(path, dtype, attn_implementation):
        config = Qwen3_5Model.config_class.from_pretrained(path)
        with torch.device("cuda:0"):
            model = Qwen3_5Model._from_config(config, dtype=dtype, attn_implementation=attn_implementation)
        state = model.state_dict()
        mapping = json.loads((Path(path) / "model.safetensors.index.json").read_text())["weight_map"]
        if set(state) != set(mapping):
            raise RuntimeError(f"Checkpoint keys mismatch: {set(state) ^ set(mapping)}")
        with torch.no_grad():
            for shard in sorted(set(mapping.values())):
                with safe_open(str(Path(path) / shard), framework="pt", device="cpu") as f:
                    for key in f.keys():
                        assert mapping[key] == shard and list(state[key].shape) == f.get_slice(key).get_shape()
                        state[key].copy_(f.get_tensor(key))
        return model
    with patch.object(Qwen3_5Model, "from_pretrained", load_cuda_checkpoint):
        decider = Decider.from_pretrained(MODEL, device="cuda", fast=False,
                                      max_length=args.max_length, batch_tokens=args.batch_tokens)
    base_state = load_file(str(MODEL / "readout.safetensors"), device="cpu")
    trained_state = load_file(str(after_path), device="cpu")
    if set(base_state) != {"weight"} or set(trained_state) != {"weight"}:
        raise ValueError("Unexpected readout tensor layout")
    if base_state["weight"].shape != (255, 5120) or trained_state["weight"].shape != base_state["weight"].shape:
        raise ValueError("Unexpected readout dimensions")
    # Original head artifact is BF16; promote both heads to FP32 for a matched comparison.
    decider.readout = torch.nn.Linear(5120, 255, bias=False, dtype=torch.float32, device="cuda")
    decider.readout.load_state_dict(base_state, strict=True)
    decider.readout.register_forward_pre_hook(lambda module, args: (args[0].float(),))
    base_weight = decider.readout.weight.detach()
    trained_weight = trained_state["weight"].to(device="cuda", dtype=torch.float32)
    decider.backbone.requires_grad_(False).eval(); decider.readout.requires_grad_(False).eval()

    # Validate this custom dual-head gather against the package's exact single-head API,
    # both without padding and with a left-padded batch of two different lengths.
    first_a = queries[0]
    check_indices = [b for b in range(len(keys)) if b != first_a][:2]
    check_requests = [request(first_a, b, texts) for b in check_indices]
    check_chunks = [decider.encoder.plan(req, args.max_length)[0] for req in check_requests]
    with torch.inference_mode():
        check_base, _ = official_marker_logits(decider, check_chunks, base_weight, trained_weight, left_pad)
        official = decider.logits(check_requests)
    observed = np.asarray([v["similar"] for v in official], dtype=np.float64)
    error = float(np.max(np.abs(np.asarray(check_base, dtype=np.float64) - observed)))
    if not np.allclose(check_base, observed, atol=1e-5, rtol=1e-5):
        raise RuntimeError(f"Dual-head reference diverges from official logits; max error={error}")
    atomic_json(out / "reference_check.json", {"status": "pass", "batch_size": len(check_chunks),
        "max_abs_error": error, "option_order": ["false", "true"],
        "same_hidden_for_both_heads": True})

    base_matrix = np.full((len(keys), len(keys)), np.nan, dtype=np.float64)
    full_matrix = np.full_like(base_matrix, np.nan)
    started = time.monotonic()
    for qi, a in enumerate(queries):
        requests = [request(a, b, texts) for b in range(len(keys)) if b != a]
        pairs = [(b, decider.encoder.plan(req, args.max_length)[0]) for b, req in
                 zip((b for b in range(len(keys)) if b != a), requests, strict=True)]
        base_values: list[float] = [math.nan] * len(requests)
        full_values: list[float] = [math.nan] * len(requests)
        # Decider's length-sorted private batching uses its configured padded-token budget.
        for batch in decider._batches([(i, chunk) for i, (_, chunk) in enumerate(pairs)]):
            chunks = [chunk for _, chunk in batch]
            with torch.inference_mode():
                base_logits, full_logits = official_marker_logits(decider, chunks, base_weight,
                                                                   trained_weight, left_pad)
            for (local_i, _), base_logit, full_logit in zip(batch, base_logits, full_logits, strict=True):
                if len(base_logit) != 2 or len(full_logit) != 2:
                    raise ValueError("Expected false/true logits")
                if not np.isfinite(base_logit).all() or not np.isfinite(full_logit).all():
                    raise FloatingPointError("Nonfinite external ranking logit")
                base_values[local_i] = float(base_logit[1] - base_logit[0])
                full_values[local_i] = float(full_logit[1] - full_logit[0])
        candidates = [b for b in range(len(keys)) if b != a]
        if not np.isfinite(base_values).all() or not np.isfinite(full_values).all():
            raise ValueError(f"Incomplete candidate row for {keys[a]}")
        base_matrix[a, candidates] = base_values
        full_matrix[a, candidates] = full_values
        np.savez_compressed(out / f"row_{a:03d}.npz", base=np.asarray(base_values), full=np.asarray(full_values),
                            candidates=np.asarray(candidates, dtype=np.int32))
        atomic_json(out / "status.json", {"status": "running", "query": qi + 1,
            "queries_total": len(queries), "key": keys[a], "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()})

    base_result = summarize(data, keys, base_matrix)
    full_result = summarize(data, keys, full_matrix)
    historical = json.loads(HISTORICAL.read_text(encoding="utf-8"))
    result = {
        "status": "complete", "base_fp32_reference": base_result, "trained_full_head": full_result,
        "trained_minus_base": {k: full_result["summary"][k] - base_result["summary"][k]
                               for k in ("top5_share", "top30_share", "directed_rank_mean",
                                         "directed_rank_median", "directed_rank_max",
                                         "last_correct_rank_mean", "last_correct_rank_median",
                                         "last_correct_rank_max")},
        "historical_bf16_base_reference": historical["summary"],
        "historical_mean_semantics": {"3.1667": "mean over 60 directed positive ranks",
                                      "3.9143": "mean over 35 query-level last-correct-positive ranks"},
        "manifest": manifest,
    }
    atomic_json(out / "comparison.json", result)
    atomic_json(out / "status.json", {"status": "complete", "queries": len(queries),
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()})
    print(json.dumps({"status": "complete", "base": base_result["summary"],
                      "trained": full_result["summary"], "delta": result["trained_minus_base"]},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()



