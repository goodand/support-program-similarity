"""학습된 LoRA adapter로 요청 파일의 텍스트를 인코딩(학습 없음). 컨테이너 안에서 실행. 평가 세션의 인코딩 요청(JSON) 형식을 그대로 받는다.
env: REQ=<요청 json, 컨테이너 경로>  REQ_SHA=<요청 json sha256>  TEXTS=<텍스트 jsonl 컨테이너 경로>  ADAPTER=<adapter 컨테이너 경로>  OUT=<출력 npz 컨테이너 경로>
인코딩 설정은 qwen_simcse.py와 동일: bf16, sdpa, max_seq_length 2048, 프롬프트 없음, L2 정규화, 길이순 배치.
중단 후 재개: 4096개마다 <OUT>.parts/에 조각 저장, 다시 실행하면 있는 조각은 건너뜀. 기존 파일은 덮어쓰지 않음(OUT이 있으면 중단).
"""
import hashlib, json, os, time
from pathlib import Path
import numpy as np, torch
from sentence_transformers import SentenceTransformer

sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
REQ, OUT = Path(os.environ["REQ"]), Path(os.environ["OUT"])
assert sha(REQ) == os.environ["REQ_SHA"], "요청 파일 해시 불일치"
assert not OUT.exists(), f"{OUT} 이미 있음(덮어쓰지 않음)"
req = json.load(open(REQ, encoding="utf-8"))
TEXTS = Path(os.environ["TEXTS"]); ADAPTER = Path(os.environ["ADAPTER"]) if os.environ.get("ADAPTER") else None  # ADAPTER 비우면 base(학습 없음) 인코딩
assert sha(TEXTS) == req["texts_file_sha256"], "텍스트 파일 해시 불일치"
want = req["hashes_to_encode"] if "hashes_to_encode" in req else None  # 없으면 텍스트 파일 전체(열 단독 요청서 형식)
if want is not None: assert len(want) == req["n_hashes"]
tx = {}
for l in open(TEXTS, encoding="utf-8"):
    r = json.loads(l)
    if r["h"] in tx: continue
    tx[r["h"]] = r["t"]
if want is None: want = list(tx); assert len(want) == req.get("n_texts", len(want)), "텍스트 수 불일치"
miss = [h for h in want if h not in tx]; assert not miss, f"텍스트 없음 {len(miss)}건"
if int(os.environ.get("ENC_LIMIT", "0")): want = want[:int(os.environ["ENC_LIMIT"])]  # 스모크용
order = sorted(range(len(want)), key=lambda i: len(tx[want[i]]))
torch.cuda.set_per_process_memory_fraction(float(os.environ.get("MEM_FRAC", "0.4")))  # 다른 학습과 GPU를 나눠 씀
m = SentenceTransformer("/models/Qwen__Qwen3-Embedding-8B", device="cuda", local_files_only=True, model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"})
if ADAPTER is not None:
  # load_adapter는 키 접두(base_model.model.)가 안 맞아 가중치를 버린다(11-05 실측: 전부 UNEXPECTED/MISSING). 직접 붙이고 전부 맞는지 확인
  from peft import LoraConfig, TaskType
  from safetensors.torch import load_file
  ac = json.load(open(ADAPTER / "adapter_config.json", encoding="utf-8"))
  m.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=ac["r"], lora_alpha=ac["lora_alpha"], lora_dropout=ac["lora_dropout"], target_modules=ac["target_modules"]))
  sd = {k.removeprefix("base_model.model.").replace(".lora_A.weight", ".lora_A.default.weight").replace(".lora_B.weight", ".lora_B.default.weight"): v
        for k, v in load_file(str(ADAPTER / "adapter_model.safetensors")).items()}  # 파일: base_model.model.<..>.lora_A.weight / 모델: <..>.lora_A.default.weight
  P = {k: v for k, v in m[0].auto_model.named_parameters() if "lora_" in k}
  assert set(P) == set(sd), f"adapter 키 불일치: 모델 {len(P)} / 파일 {len(sd)} / 공통 {len(set(P) & set(sd))}"
  with torch.no_grad():
      for k, v in sd.items(): P[k].copy_(v.to(P[k].dtype))
m.eval(); m.max_seq_length = req["encode_max_seq_length"]
# 검증: 같은 adapter로 학습 직후 만든 qwen_chunk.npz의 앞 32건을 다시 인코딩해 같은지 확인(adapter가 실제로 적용됐는지)
ref = ADAPTER.parent / "qwen_chunk.npz" if ADAPTER is not None else None
checked = ref is not None and ref.exists() and ref != OUT  # 시작 시점에 판정(11-06: OUT이 ref와 같은 경로면 저장 후 존재해 manifest에서 NameError)
if checked:
    rows = [json.loads(l) for _, l in zip(range(32), open("/data/encode_all_v1.jsonl", encoding="utf-8"))]
    with np.load(ref) as z: e = dict(zip(z["h"][:2000].tolist(), z["v"][:2000]))
    rows = [r for r in rows if r["sha256"] in e]
    with torch.no_grad(): v0 = m.encode([r["text"] for r in rows], batch_size=8, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    cos = (v0 * np.stack([e[r["sha256"]] for r in rows])).sum(1); print(f"검증 {len(rows)}건 cos min {cos.min():.5f} mean {cos.mean():.5f}", flush=True)
    assert len(rows) >= 16 and cos.min() > 0.99, "adapter 적용 검증 실패"  # 실측: adapter 적용 시 0.998(bf16 배치 잡음), base vs crop은 0.80~0.88
parts = Path(str(OUT) + ".parts"); parts.mkdir(parents=True, exist_ok=True)
t0 = time.time(); S = 4096
for s in range(0, len(order), S):
    p = parts / f"{s:06d}.npy"
    if p.exists(): continue
    idx = order[s:s + S]
    with torch.no_grad():
        v = m.encode([tx[want[i]] for i in idx], batch_size=int(os.environ.get("BATCH", "16")), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
    np.save(p, v.astype(np.float32)); torch.cuda.empty_cache()
    print(f"enc {min(s + S, len(order))}/{len(order)} {time.time() - t0:.0f}s", flush=True)
V = np.zeros((len(want), 4096), np.float32)
for s in range(0, len(order), S): V[order[s:s + S]] = np.load(parts / f"{s:06d}.npy")
np.savez(OUT, h=np.array(want), v=V)
man = {"out": str(OUT), "npz_sha256": sha(OUT), "n": len(want), "request": str(REQ), "request_sha256": os.environ["REQ_SHA"], "texts_sha256": req["texts_file_sha256"],
       "adapter": str(ADAPTER) if ADAPTER else None, "adapter_sha256": sha(ADAPTER / "adapter_model.safetensors") if ADAPTER else None, "encode_max_seq_length": m.max_seq_length, "prompt": None, "check_vs_qwen_chunk_cos_min": float(cos.min()) if checked else None,
       "dtype": "float32", "l2_normalized": True, "encode_seconds_this_run": round(time.time() - t0, 1), "image": "embed-ft:20261102-v1", "gpu": torch.cuda.get_device_name(0)}
Path(str(OUT) + ".manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print("RESULT", json.dumps(man, ensure_ascii=False), flush=True)
