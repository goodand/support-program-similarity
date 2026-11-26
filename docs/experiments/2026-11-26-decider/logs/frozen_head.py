"""Tokenize, benchmark, cache frozen-backbone features, and train a readout head.

All model forwards use pplx_decider's official Encoder, reference Qwen3_5Model
implementation, exact noncausal full-attention mask hook, left-padding collator,
and recurrent linear-attention mask. This file does not launch unless explicitly
called with a GPU subcommand by the coordinating agent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import struct
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file, save_file
from transformers import AutoProcessor
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model


HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "decider_identity_v1"
MODEL = Path(r"D:/dev_eval_team/_models_harness/pplx-decider-v1.2")
MODEL_SRC = MODEL / "source" / "src"
PYTHON = Path(r"D:/dev_eval_team/_decider_env_harness/Scripts/python.exe")
MAX_LENGTH = 2048
torch.set_num_threads(8)


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=True).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def load_package() -> tuple[Any, Any, Any, int, int, Any]:
    sys.path.insert(0, str(MODEL_SRC))
    from pplx_decider.model import enable_noncausal_full_attention
    from pplx_decider.protocol import Encoded, Encoder, MAX_OPTIONS, MIN_PIXELS, left_pad
    return enable_noncausal_full_attention, Encoder, Encoded, MAX_OPTIONS, MIN_PIXELS, left_pad


def check_inputs() -> dict[str, Any]:
    manifest_path = DATA / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        actual = digest_file(DATA / name)
        if actual != expected:
            raise ValueError(f"Prepared dataset hash mismatch for {name}: {actual}")
    return {"manifest": manifest, "manifest_path": manifest_path,
            "manifest_sha256": digest_file(manifest_path),
            "train_sha256": digest_file(DATA / "train.jsonl"),
            "validation_sha256": digest_file(DATA / "validation.jsonl")}


def rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def prepare_split(path: Path, out: Path, split: str, encoder: Any) -> dict[str, Any]:
    token_path = out / f"{split}.tokens.u32"
    meta_path = out / f"{split}.rows.jsonl"
    offsets, lengths, markers, labels, weights = [0], [], [], [], []
    n = 0
    started = time.monotonic()
    with token_path.open("wb") as token_file, meta_path.open("w", encoding="utf-8") as meta_file:
        for row in rows(path):
            plan = encoder.plan(row["request"], MAX_LENGTH)
            if len(plan) != 1:
                raise ValueError(f"Unexpected multi-chunk row: {row['id']}")
            chunk = plan[0]
            if chunk.length > MAX_LENGTH:
                raise ValueError(f"Token limit exceeded ({chunk.length}) by {row['id']}")
            ids = chunk.input_ids.numpy().astype("<u4", copy=False)
            ids.tofile(token_file)
            offsets.append(offsets[-1] + len(ids))
            lengths.append(len(ids)); markers.append(int(chunk.markers[0]))
            labels.append(int(bool(row["target"])))
            weights.append(float(row.get("sample_weight", 1.0)))
            meta_file.write(json.dumps({k: row[k] for k in
                ("id", "target", "label_source", "a_document_id", "b_document_id", "a_group", "b_group")
                if k in row}, ensure_ascii=False) + "\n")
            n += 1
            if n % 5000 == 0:
                print(f"tokenized {split}: {n}", flush=True)
    arrays = {
        "offsets": np.asarray(offsets, dtype="<i8"),
        "lengths": np.asarray(lengths, dtype="<i4"),
        "markers": np.asarray(markers, dtype="<i4"),
        "labels": np.asarray(labels, dtype="i1"),
        "weights": np.asarray(weights, dtype="<f4"),
    }
    for name, array in arrays.items():
        np.save(out / f"{split}.{name}.npy", array, allow_pickle=False)
    if n != len(lengths) or n != len(labels) or offsets[-1] * 4 != token_path.stat().st_size:
        raise AssertionError("Prepared token cache count or byte size mismatch")
    return {"rows": n, "token_count": int(offsets[-1]), "max_length": int(max(lengths)),
            "mean_length": float(np.mean(lengths)), "tokenized_rows_per_sec": n / (time.monotonic()-started),
            "files": {p.name: digest_file(p) for p in out.glob(f"{split}.*") if p.is_file()}}


def cmd_prepare(args: argparse.Namespace) -> None:
    input_info = check_inputs()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    enable_noncausal, Encoder, _, _, min_pixels, _ = load_package()
    processor = AutoProcessor.from_pretrained(str(MODEL))
    processor.tokenizer.padding_side = "left"
    processor.image_processor.size = {"shortest_edge": min_pixels,
        "longest_edge": int(json.loads((MODEL / "decision_config.json").read_text(encoding="utf-8"))["max_pixels"])}
    config = json.loads((MODEL / "decision_config.json").read_text(encoding="utf-8"))
    encoder = Encoder(processor, config["codes"])
    info = {"created_utc": datetime.now(timezone.utc).isoformat(), "source_dir": str(DATA),
            "manifest_sha256": input_info["manifest_sha256"], "train_sha256": input_info["train_sha256"],
            "validation_sha256": input_info["validation_sha256"], "model_dir": str(MODEL),
            "tokenizer_revision": config.get("revision"), "max_length": MAX_LENGTH,
            "tokenizer": "pplx_decider.protocol.Encoder, per-row exact encoding; no truncation",
            "split": {}}
    for split in ("train", "validation"):
        info["split"][split] = prepare_split(DATA / f"{split}.jsonl", out, split, encoder)
    info["cache_sha256"] = {p.name: digest_file(p) for p in out.iterdir() if p.is_file()}
    atomic_json(out / "token_cache_manifest.json", info)
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


def open_cache(cache: Path, split: str) -> dict[str, Any]:
    info = json.loads((cache / "token_cache_manifest.json").read_text(encoding="utf-8"))
    for name, expected in info["cache_sha256"].items():
        if digest_file(cache / name) != expected:
            raise ValueError(f"Token cache hash mismatch: {name}")
    meta = list(rows(cache / f"{split}.rows.jsonl"))
    arrays = {key: np.load(cache / f"{split}.{key}.npy", mmap_mode="r", allow_pickle=False)
              for key in ("offsets", "lengths", "markers", "labels", "weights")}
    ids = np.memmap(cache / f"{split}.tokens.u32", mode="r", dtype="<u4")
    if len(meta) != len(arrays["labels"]) or len(meta) != len(arrays["lengths"]):
        raise ValueError("Token cache row count mismatch")
    arrays.update(rows=meta, ids=ids, info=info)
    return arrays


def encoded_at(data: dict[str, Any], index: int, Encoded: Any) -> Any:
    start, end = int(data["offsets"][index]), int(data["offsets"][index+1])
    token_ids = torch.from_numpy(np.asarray(data["ids"][start:end], dtype=np.int64).copy())
    return Encoded(keys=("same_program",), counts=(2,), input_ids=token_ids,
                   markers=torch.tensor([int(data["markers"][index])], dtype=torch.long))


def batches(lengths: np.ndarray, indices: list[int], budget: int) -> Iterator[list[int]]:
    ordered = sorted(indices, key=lambda i: int(lengths[i]))
    batch: list[int] = []
    max_len = 0
    for index in ordered:
        length = int(lengths[index])
        if batch and (len(batch) + 1) * max(max_len, length) > budget:
            yield batch
            batch, max_len = [], 0
        batch.append(index)
        max_len = max(max_len, length)
    if batch:
        yield batch


def load_backbone(device: torch.device) -> tuple[torch.nn.Module, Any, Any]:
    enable_noncausal, Encoder, Encoded, n_options, min_pixels, left_pad = load_package()
    sizes = (n_options, min_pixels)
    torch.backends.cuda.enable_cudnn_sdp(False)
    model = Qwen3_5Model.from_pretrained(str(MODEL), dtype=torch.bfloat16, attn_implementation="sdpa")
    enable_noncausal(model.language_model)
    model.requires_grad_(False).eval().to(device)
    return model, (Encoder, Encoded, sizes), left_pad


@torch.inference_mode()
def forward_features(model: torch.nn.Module, items: list[tuple[int, Any]], left_pad: Any,
                     pad_id: int, device: torch.device) -> list[torch.Tensor]:
    chunks = [chunk for _, chunk in items]
    inputs = {k: v.to(device) for k, v in left_pad(chunks, pad_id).items()}
    hidden = model(**inputs, use_cache=False).last_hidden_state
    offsets = [inputs["input_ids"].shape[1]-c.length for c in chunks]
    rows = torch.arange(len(chunks), device=device)
    cols = torch.tensor([int(c.markers[0]) + offsets[i] for i, c in enumerate(chunks)], device=device)
    return list(hidden[rows, cols].to(torch.bfloat16).cpu().unbind(0))


def cmd_benchmark(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    if args.rows < 1 or not args.budgets or any(b < 1 for b in args.budgets):
        raise ValueError("Benchmark rows and token budgets must be positive")
    cache = args.cache.resolve(); data = open_cache(cache, args.split)
    _, _, Encoded, _, _, left_pad = load_package()
    all_ids = list(range(len(data["lengths"])))
    sample_ids = sorted(all_ids, key=lambda i: hashlib.sha256(f"{args.seed}:{i}".encode()).hexdigest())[:args.rows]
    device = torch.device("cuda")
    processor = AutoProcessor.from_pretrained(str(MODEL))
    pad_id = int(processor.tokenizer.pad_token_id)
    model, _, _ = load_backbone(device)
    results = []
    for budget in args.budgets:
        selected_batches = list(batches(data["lengths"], sample_ids, budget))
        # Warm one batch and measure the fixed sample; catch larger budgets that do not fit.
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device); started = time.perf_counter(); measured = 0
        try:
            if selected_batches:
                warm = selected_batches[0]
                forward_features(model, [(i, encoded_at(data, i, Encoded)) for i in warm], left_pad, pad_id, device)
            for batch in selected_batches:
                forward_features(model, [(i, encoded_at(data, i, Encoded)) for i in batch], left_pad, pad_id, device)
                measured += len(batch)
        except torch.cuda.OutOfMemoryError as error:
            torch.cuda.empty_cache()
            results.append({"token_budget": budget, "status": "out_of_memory", "rows_completed_before_failure": measured,
                            "error": str(error), "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
                            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device)})
            continue
        torch.cuda.synchronize(device); elapsed = time.perf_counter()-started
        results.append({"token_budget": budget, "rows": measured, "batches": len(selected_batches),
            "elapsed_sec": elapsed, "rows_per_sec": measured/elapsed if elapsed else None,
            "corpus_rows": len(data["lengths"]), "estimated_corpus_hours": len(data["lengths"])/max(measured/elapsed,1e-12)/3600,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "mean_padded_batch_tokens": float(np.mean([len(b)*max(int(data['lengths'][i]) for i in b) for b in selected_batches])) if selected_batches else 0})
    out = args.out.resolve()
    if out.exists(): raise FileExistsError(out)
    atomic_json(out, {"created_utc": datetime.now(timezone.utc).isoformat(), "cache": str(cache),
        "split": args.split, "sample_rows": args.rows, "seed": args.seed, "results": results,
        "semantics": "official padded reference forward; BF16 frozen backbone; marker features only"})
    print(json.dumps(results, indent=2), flush=True)


def cmd_cache_features(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required")
    if args.batch_tokens < 1: raise ValueError("batch-tokens must be positive")
    token_cache, out = args.cache.resolve(), args.out.resolve()
    if not token_cache.is_dir(): raise FileNotFoundError(token_cache)
    token_manifest_hash = digest_file(token_cache/"token_cache_manifest.json")
    enable_noncausal, Encoder, Encoded, _, _, left_pad = load_package()
    processor = AutoProcessor.from_pretrained(str(MODEL)); pad_id = int(processor.tokenizer.pad_token_id)
    device = torch.device("cuda"); model, _, _ = load_backbone(device)
    hidden_size = int(model.config.text_config.hidden_size)
    config = {"token_cache": str(token_cache), "token_cache_manifest_sha256": token_manifest_hash,
        "model_dir": str(MODEL), "model_config_sha256": digest_file(MODEL/"config.json"),
        "decision_config_sha256": digest_file(MODEL/"decision_config.json"), "hidden_size": hidden_size,
        "dtype": "bfloat16", "batch_tokens": args.batch_tokens, "max_length": MAX_LENGTH,
        "padding": "official left_pad", "attention": "official Decider reference hook"}
    manifest_path = out/"feature_cache_manifest.json"
    if args.resume:
        if not out.is_dir() or not manifest_path.exists():
            raise FileNotFoundError("Feature-cache resume requires an existing manifest")
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if any(old.get(k) != v for k,v in config.items()):
            raise ValueError("Feature-cache resume settings differ")
        feature_manifest = old
    else:
        out.mkdir(parents=True, exist_ok=False)
        feature_manifest = {**config, "created_utc": datetime.now(timezone.utc).isoformat(), "splits": {}}
        atomic_json(manifest_path, feature_manifest)
    for split in ("train", "validation"):
        data = open_cache(token_cache, split); n = len(data["lengths"])
        feature_path = out/f"{split}.features.bf16"
        done_path = out/f"{split}.features_done.u1"
        if args.resume and feature_path.exists() and done_path.exists():
            features = np.memmap(feature_path, mode="r+", dtype="<u2", shape=(n, hidden_size))
            done = np.memmap(done_path, mode="r+", dtype="u1", shape=(n,))
            if np.any(done > 1): raise ValueError(f"Invalid completion bitmap for {split}")
            recorded = feature_manifest.get("splits", {}).get(split, {}).get("completed_rows", 0)
            if int(done.sum()) < int(recorded): raise ValueError(f"Completion bitmap regressed for {split}")
        elif args.resume and (feature_path.exists() or done_path.exists()):
            raise FileNotFoundError(f"Only one of the feature/done files exists for {split}")
        else:
            features = np.memmap(feature_path, mode="w+", dtype="<u2", shape=(n, hidden_size))
            done = np.memmap(done_path, mode="w+", dtype="u1", shape=(n,)); done[:] = 0; done.flush()
            feature_manifest["splits"][split] = {"rows": n, "features_file": feature_path.name,
                "done_file": done_path.name, "completed_rows": 0, "complete": False}
        indices = list(range(n)); started = time.monotonic(); total=0
        for batch_num, idxs in enumerate(batches(data["lengths"], indices, args.batch_tokens), 1):
            todo = [i for i in idxs if not done[i]]
            if todo:
                items = [(i, encoded_at(data, i, Encoded)) for i in todo]
                vectors = forward_features(model, items, left_pad, pad_id, device)
                for i, vec in zip(todo, vectors):
                    features[i] = vec.contiguous().view(torch.uint16).numpy()
                    done[i] = 1
                features.flush(); done.flush()
            total += len(todo)
            if batch_num % 100 == 0:
                feature_manifest["splits"][split]["completed_rows"] = int(done.sum())
                atomic_json(manifest_path, feature_manifest)
                print(f"cached {split}: {int(done.sum())}/{n} rows; {int(done.sum())/max(time.monotonic()-started,1e-9):.2f} rows/sec", flush=True)
        features.flush(); done.flush()
        complete = int(done.sum())
        if complete != n: raise AssertionError(f"Incomplete {split} features: {complete}/{n}")
        feature_manifest["splits"][split].update({"completed_rows": complete, "complete": True,
            "features_sha256": digest_file(feature_path), "rows_per_sec": total/max(time.monotonic()-started,1e-9)})
        atomic_json(manifest_path, feature_manifest)
    feature_manifest["completed_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_json(manifest_path, feature_manifest)
    print(json.dumps(feature_manifest, indent=2), flush=True)

def eval_head(features: np.memmap, labels: np.ndarray, weights: np.ndarray,
              weight: torch.Tensor, device: torch.device, batch_size: int=512) -> dict[str, float]:
    loss_sum = 0.0; weighted_count = 0.0; correct = {0: 0, 1: 0}; count = {0: 0, 1: 0}
    for start in range(0, len(labels), batch_size):
        end = min(start+batch_size, len(labels)); ids = np.arange(start, end)
        x_bf16 = torch.from_numpy(np.asarray(features[ids]).copy()).view(torch.bfloat16).to(device)
        x = x_bf16.float()
        y = torch.as_tensor(labels[ids].astype(np.int64), device=device)
        w = torch.as_tensor(weights[ids].astype(np.float32), device=device)
        logits = F.linear(x, weight[:2].float())
        loss = F.cross_entropy(logits, y, reduction="none")
        loss_sum += float((loss*w).sum()); weighted_count += float(w.sum())
        pred = logits.argmax(-1)
        for label in (0, 1):
            mask = y==label; count[label] += int(mask.sum()); correct[label] += int(((pred==y)&mask).sum())
    recall0, recall1 = correct[0]/count[0], correct[1]/count[1]
    return {"loss": loss_sum/weighted_count, "accuracy": sum(correct.values())/sum(count.values()),
        "negative_recall": recall0, "positive_recall": recall1, "balanced_accuracy": (recall0+recall1)/2,
        "rows": len(labels)}


def cmd_train(args: argparse.Namespace) -> None:
    token_cache, feature_dir = args.cache.resolve(), args.features.resolve()
    feature_info = json.loads((feature_dir/"feature_cache_manifest.json").read_text(encoding="utf-8"))
    if feature_info["token_cache_manifest_sha256"] != digest_file(token_cache/"token_cache_manifest.json"):
        raise ValueError("Features and token cache manifest do not match")
    for split in ("train", "validation"):
        entry = feature_info.get("splits", {}).get(split, {})
        if not entry.get("complete") or digest_file(feature_dir/entry["features_file"]) != entry.get("features_sha256"):
            raise ValueError(f"Feature cache is incomplete or corrupt for {split}")
    out = args.out.resolve()
    if args.resume is None:
        out.mkdir(parents=True, exist_ok=False)
    elif not out.is_dir() or args.resume.resolve().parent != out:
        raise ValueError("Resume requires an existing output directory and checkpoint inside it")
    tr = open_cache(token_cache, "train"); va = open_cache(token_cache, "validation")
    train_features_path = feature_dir/feature_info["splits"]["train"]["features_file"]
    val_features_path = feature_dir/feature_info["splits"]["validation"]["features_file"]
    h = int(feature_info["hidden_size"]); n = len(tr["labels"])
    train_features = np.memmap(train_features_path, mode="r", dtype="<u2", shape=(n,h))
    nv = len(va["labels"]); val_features = np.memmap(val_features_path, mode="r", dtype="<u2", shape=(nv,h))
    head = torch.nn.Linear(h, 255, bias=False, dtype=torch.float32)
    original = load_file(str(MODEL/"readout.safetensors"), device="cpu")
    head.load_state_dict(original, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head.to(device); base_weight = head.weight.detach().clone()
    optimizer = torch.optim.AdamW([head.weight], lr=args.lr, weight_decay=0.0)
    config = {"rows": n, "validation_rows": nv, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "seed": args.seed, "token_cache_manifest_sha256": digest_file(token_cache/"token_cache_manifest.json"),
        "feature_cache_manifest_sha256": digest_file(feature_dir/"feature_cache_manifest.json"),
        "train_features_sha256": feature_info["splits"]["train"]["features_sha256"],
        "validation_features_sha256": feature_info["splits"]["validation"]["features_sha256"],
        "base_readout_sha256": digest_file(MODEL/"readout.safetensors"), "base_weights_saved": False}
    config_path = out/"head_training.json"
    rng = np.random.default_rng(args.seed); epoch=0; cursor=0; permutation=rng.permutation(n).astype(np.int64)
    if args.resume is None:
        atomic_json(config_path, config)
        save_file({"weight": head.weight.detach().cpu().contiguous()}, out/"readout_before.safetensors")
        initial_checkpoint = out/"head-checkpoint-e01-r000000.pt"
        if initial_checkpoint.exists(): raise FileExistsError(initial_checkpoint)
        temp_checkpoint = initial_checkpoint.with_suffix(".pt.tmp")
        torch.save({"epoch": epoch, "cursor": cursor, "permutation": torch.from_numpy(permutation),
            "head_weight": head.weight.detach().cpu(), "optimizer": optimizer.state_dict(),
            "numpy_rng": rng.bit_generator.state, "config": config}, temp_checkpoint)
        temp_checkpoint.replace(initial_checkpoint)
    else:
        old = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("rows", "validation_rows", "epochs", "batch_size", "lr", "seed",
                    "token_cache_manifest_sha256", "feature_cache_manifest_sha256", "train_features_sha256",
                    "validation_features_sha256", "base_readout_sha256"):
            if old.get(key) != config.get(key): raise ValueError(f"Resume setting changed: {key}")
        saved_base = load_file(str(out/"readout_before.safetensors"), device="cpu")["weight"]
        if not torch.equal(saved_base, original["weight"]):
            raise ValueError("Saved initial readout differs from the local v1.2 checkpoint")
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("config") != config:
            raise ValueError("Checkpoint and run configuration differ")
        head.load_state_dict({"weight": state["head_weight"]}, strict=True)
        optimizer.load_state_dict(state["optimizer"]); epoch=int(state["epoch"])
        cursor=int(state["cursor"]); permutation=state["permutation"].numpy()
        rng.bit_generator.state=state["numpy_rng"]
    val_before = eval_head(val_features, va["labels"], va["weights"], base_weight, device)
    log_path=out/"head_training.jsonl"
    def log(obj: dict[str,Any]) -> None:
        with log_path.open("a",encoding="utf-8") as f: f.write(json.dumps(obj,ensure_ascii=True)+"\n")
        print(json.dumps(obj,ensure_ascii=True),flush=True)
    if args.resume is None: log({"event":"base_validation","metrics":val_before})
    batches_per_epoch=math.ceil(n/args.batch_size); global_step=epoch*batches_per_epoch+math.ceil(cursor/args.batch_size)
    total_started=time.monotonic(); interval_started=total_started; rows_since_save=0
    while epoch < args.epochs:
        if len(permutation)!=n or len(np.unique(permutation))!=n:
            raise ValueError("Training permutation is not a unique full-coverage ordering")
        head.train()
        while cursor < n:
            idx=permutation[cursor:min(cursor+args.batch_size,n)]
            x=torch.from_numpy(np.asarray(train_features[idx]).copy()).view(torch.bfloat16).to(device).float()
            y=torch.as_tensor(tr["labels"][idx].astype(np.int64),device=device)
            w=torch.as_tensor(tr["weights"][idx].astype(np.float32),device=device)
            optimizer.zero_grad(set_to_none=True)
            logits=F.linear(x,head.weight[:2].float())
            losses=F.cross_entropy(logits,y,reduction="none")
            loss=(losses*w).sum()/w.sum().clamp_min(1e-12)
            if not torch.isfinite(loss): raise FloatingPointError(f"Nonfinite loss at step {global_step+1}")
            loss.backward()
            if head.weight.grad is None or not torch.isfinite(head.weight.grad).all():
                raise FloatingPointError(f"Nonfinite head gradient at step {global_step+1}")
            grad_norm=torch.nn.utils.clip_grad_norm_([head.weight],args.grad_clip)
            if not torch.isfinite(grad_norm): raise FloatingPointError("Nonfinite gradient norm")
            optimizer.step()
            cursor += len(idx); global_step += 1; rows_since_save += len(idx)
            if global_step % 25 == 0 or cursor == n:
                log({"event":"head_step","epoch":epoch+1,"step":global_step,"rows_seen_epoch":cursor,
                     "epoch_coverage":cursor/n,"loss":float(loss.detach()),"gradient_norm":float(grad_norm),
                     "rows_per_sec":rows_since_save/max(time.monotonic()-interval_started,1e-9)})
            if cursor == n or cursor % args.save_every_rows < len(idx):
                ckpt=out/f"head-checkpoint-e{epoch+1:02d}-r{cursor:06d}.pt"
                if ckpt.exists(): raise FileExistsError(f"Refusing to overwrite {ckpt}")
                tmp=ckpt.with_suffix(".pt.tmp")
                torch.save({"epoch":epoch,"cursor":cursor,"permutation":torch.from_numpy(permutation),
                    "head_weight":head.weight.detach().cpu(),"optimizer":optimizer.state_dict(),
                    "numpy_rng":rng.bit_generator.state,"config":config},tmp)
                tmp.replace(ckpt); rows_since_save=0; interval_started=time.monotonic()
        if cursor != n: raise AssertionError("Epoch ended before full coverage")
        log({"event":"epoch_coverage_complete","epoch":epoch+1,"rows_seen":cursor,"unique_rows":len(np.unique(permutation)),
             "expected_rows":n,"coverage":1.0})
        epoch += 1
        cursor=0
        if epoch < args.epochs: permutation=rng.permutation(n).astype(np.int64)
        # Save an epoch boundary so an interruption cannot lose the full-pass marker.
        ckpt=out/f"head-checkpoint-e{epoch:02d}-complete.pt"
        if ckpt.exists(): raise FileExistsError(f"Refusing to overwrite {ckpt}")
        tmp=ckpt.with_suffix(".pt.tmp")
        torch.save({"epoch":epoch,"cursor":cursor,"permutation":torch.from_numpy(permutation),
            "head_weight":head.weight.detach().cpu(),"optimizer":optimizer.state_dict(),
            "numpy_rng":rng.bit_generator.state,"config":config},tmp); tmp.replace(ckpt)
    val_after=eval_head(val_features,va["labels"],va["weights"],head.weight,device)
    if (out/"readout_after.safetensors").exists() or (out/"head_evaluation.json").exists():
        raise FileExistsError("Refusing to overwrite completed head-evaluation artifacts")
    save_file({"weight":head.weight.detach().cpu().contiguous()},out/"readout_after.safetensors")
    result={"before":val_before,"after":val_after,"completed_epochs":epoch,"train_rows_per_epoch":n,
        "full_coverage":epoch>=1,"train_rows_unique_per_epoch":n,"head_only":True,"base_weights_saved":False,
        "elapsed_seconds":time.monotonic()-total_started,"config":config}
    atomic_json(out/"head_evaluation.json",result)
    log({"event":"final_evaluation","result":result})


def main() -> None:
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser("prepare",help="CPU-tokenize every prepared train/validation row")
    p.add_argument("--out",type=Path,required=True); p.set_defaults(func=cmd_prepare)
    b=sub.add_parser("benchmark",help="GPU throughput/memory sweep on a fixed token-cache sample")
    b.add_argument("--cache",type=Path,required=True); b.add_argument("--out",type=Path,required=True)
    b.add_argument("--split",choices=("train","validation"),default="train"); b.add_argument("--rows",type=int,default=256)
    b.add_argument("--seed",type=int,default=123); b.add_argument("--budgets",type=int,nargs="+",default=[2048,4096,8192])
    b.set_defaults(func=cmd_benchmark)
    c=sub.add_parser("cache-features",help="GPU-cache frozen BF16 marker vectors for all examples")
    c.add_argument("--cache",type=Path,required=True); c.add_argument("--out",type=Path,required=True)
    c.add_argument("--batch-tokens",type=int,default=4096); c.add_argument("--resume",action="store_true")
    c.set_defaults(func=cmd_cache_features)
    t=sub.add_parser("train-head",help="Train one or more complete passes over unique cached features")
    t.add_argument("--cache",type=Path,required=True); t.add_argument("--features",type=Path,required=True)
    t.add_argument("--out",type=Path,required=True); t.add_argument("--epochs",type=int,default=1)
    t.add_argument("--batch-size",type=int,default=512); t.add_argument("--lr",type=float,default=1e-4)
    t.add_argument("--seed",type=int,default=17); t.add_argument("--grad-clip",type=float,default=1.0)
    t.add_argument("--save-every-rows",dest="save_every_rows",type=int,default=4096)
    t.add_argument("--resume",type=Path,default=None); t.set_defaults(func=cmd_train)
    args=parser.parse_args()
    if args.command == "train-head" and (args.epochs < 1 or args.batch_size < 1 or args.save_every_rows < 1
                                           or args.lr <= 0 or args.grad_clip <= 0):
        parser.error("epochs, batch-size, save-every-rows, learning rate and grad-clip must be positive")
    args.func(args)


if __name__=="__main__": main()
