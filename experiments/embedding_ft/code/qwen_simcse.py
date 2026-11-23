"""Qwen3-Embedding-8B + LoRA 비지도 SimCSE + 채점용 npz 인코딩. 컨테이너 안에서 실행.
SimCSE 노이즈: Qwen3 기본 attention_dropout=0 → LoRA dropout만으론 loss≈0.002(스모크)라 attention_dropout=0.1을 켠다(학습 때만 작동).
env: RUN_ID  LIMIT=<학습 텍스트 수, 0=전부>  ENC_LIMIT=<인코딩 수, 0=전부(스모크용)>
출력: /output/<RUN_ID>/{qwen_chunk.npz, manifest.json, adapter/}
인코딩 설정은 채점 기준선(embed_v2.py)과 동일: bf16, sdpa, max_seq_length 2048, 프롬프트 없음, L2 정규화.
"""
import hashlib, json, os, random, time
from pathlib import Path
import numpy as np, torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses

RUN_ID, LIMIT, ENC_LIMIT = os.environ["RUN_ID"], int(os.environ.get("LIMIT", "0")), int(os.environ.get("ENC_LIMIT", "0"))
SEED, ENC_LEN = int(os.environ.get("SEED", "42")), 2048
TRAIN_LEN = int(os.environ.get("TRAIN_LEN", "256"))  # 기본값은 v1과 동일(재현용)
CORPUS = os.environ.get("CORPUS", "train_docs_v2.jsonl")  # 기본 v2(예산서 원문 문서 단위). v1(train_corpus_v1.jsonl)은 메인 세션 chunk라 학습용 아님(사용자 지시 11-04)
HYPOTHESIS = os.environ.get("HYPOTHESIS", "").strip()
assert HYPOTHESIS, "HYPOTHESIS(실행 전 주 가설 한 줄)가 비어 있음 — 사후 가설 선택 방지. 실측용이면 '길이 실측(비교 아님)' 등으로 명시"
_r = int(os.environ.get("LORA_R", "16"))  # 기본값 = 밤 레시피. 선별 묶음 B용 env(LR·LORA_R·LORA_TARGETS·CROP_*)는 비우면 기존과 동일
HP = dict(batch=int(os.environ.get("BATCH", "64")), lr=float(os.environ.get("LR", "5e-5")), epochs=int(os.environ.get("EPOCHS", "1")), lora_r=_r, lora_alpha=2 * _r, lora_dropout=0.1, attention_dropout=0.1,
          target_modules=os.environ.get("LORA_TARGETS", "q_proj,k_proj,v_proj,o_proj").split(","),
          crop=[float(os.environ.get("CROP_MIN_FRAC", "0.1")), float(os.environ.get("CROP_MAX_FRAC", "0.5")), int(os.environ.get("CROP_MIN_CHARS", "150"))])
MODEL = "/models/Qwen__Qwen3-Embedding-8B"
out = Path("/output") / RUN_ID; out.mkdir(parents=True, exist_ok=True)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
random.seed(SEED); torch.manual_seed(SEED)

docs = [json.loads(l) for l in open("/data/" + CORPUS, encoding="utf-8")]
random.shuffle(docs)  # 길이가 같으므로 기존(텍스트 목록 섞기)과 같은 순열
corpus = [d["text"] for d in docs]; depts = [d.get("dept") for d in docs]
DEPT_BATCH = os.environ.get("DEPT_BATCH", "") == "1"  # B6: 한 묶음을 같은 부처 문서로 채움(in-batch 오답 = 같은 부처). crop 전용
if LIMIT and not os.environ.get("PAIRS_FILE"): corpus = corpus[:LIMIT]  # 짝 모드(crop 포함)에서는 짝 단계에서만 LIMIT 적용

m = SentenceTransformer(MODEL, device="cuda", local_files_only=True, model_kwargs={"torch_dtype": torch.bfloat16, "attn_implementation": "sdpa"},
                       config_kwargs={"attention_dropout": HP["attention_dropout"]})
m.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=HP["lora_r"], lora_alpha=HP["lora_alpha"],
                         lora_dropout=HP["lora_dropout"], target_modules=HP["target_modules"]))
m.max_seq_length = TRAIN_LEN
t0 = time.time()
PAIRS_FILE = os.environ.get("PAIRS_FILE", "")  # 비우면 SimCSE(자기 자신이 짝). 지정하면 그 짝 파일(anchor/positive)로 학습
MAX_STEPS = int(os.environ.get("MAX_STEPS", "0"))  # >0이면 길이·속도 실측용으로 그 스텝만
if PAIRS_FILE:
    def span(t):  # Contriever식 구간: 글자 기준, 문서의 CROP_MIN_FRAC~CROP_MAX_FRAC(기본 10~50%), 최소 CROP_MIN_CHARS(150)자. 새 문장은 만들지 않음
        n = len(t); L = max(HP["crop"][2], int(n * random.uniform(HP["crop"][0], HP["crop"][1]))); s0 = random.randint(0, max(0, n - L)); return t[s0:s0 + L]
    if PAIRS_FILE == "crop":
        prs = [{"anchor": span(t), "positive": span(t), "dept": dp} for t, dp in zip(corpus, depts) if len(t) >= 400]
        npairs_file = len(prs)
    else:
        def load_pairs(fn):  # 짝 파일 → 사업당 한 짝만(seed에 따라 다른 짝): 같은 사업의 연도 사슬 짝이 한 묶음에서 가짜 오답이 되는 것을 막음
            ps = [json.loads(l) for l in open("/data/" + fn, encoding="utf-8")]; random.shuffle(ps); one = {}
            for r in ps: one.setdefault(r.get("prog", id(r)), r)
            return len(ps), list(one.values())
        if PAIRS_FILE.startswith("crop+"):  # D7: crop 짝 + 파일 짝 1:1 혼합(각 LIMIT/2). 파일 짝에는 dept가 없어 DEPT_BATCH 불가
            fn = PAIRS_FILE[5:]; n_f, pf = load_pairs(fn); half = (LIMIT // 2) if LIMIT else min(len(pf), 10**9)
            pc = [{"anchor": span(t), "positive": span(t)} for t in corpus if len(t) >= 400]
            prs = pc[:half] + [{"anchor": r["anchor"], "positive": r["positive"]} for r in pf[:half]]; random.shuffle(prs); npairs_file = len(pc) + n_f; LIMIT = 0
        else:
            npairs_file, prs = load_pairs(PAIRS_FILE)
    if LIMIT: prs = prs[:LIMIT]
    ds = Dataset.from_dict({"anchor": [r["anchor"] for r in prs], "positive": [r["positive"] for r in prs]})
else:
    ds = Dataset.from_dict({"anchor": corpus, "positive": corpus})
args = SentenceTransformerTrainingArguments(output_dir=str(out / "ckpt"), num_train_epochs=HP["epochs"], per_device_train_batch_size=HP["batch"],
    learning_rate=HP["lr"], warmup_ratio=0.1, bf16=True, gradient_checkpointing=True, logging_steps=10,
    save_strategy="steps", save_steps=50, save_total_limit=2,  # 정확 재개용(옵티마이저·스케줄러·난수·step 포함). 끄지 말 것
    report_to=[], seed=SEED, dataloader_drop_last=True, max_steps=MAX_STEPS or -1,
    batch_sampler="no_duplicates" if PAIRS_FILE else "batch_sampler")  # 같은 글이 한 묶음에 둘 있으면 가짜 오답이 되므로 짝 학습은 중복 방지
resume = any((out / "ckpt").glob("checkpoint-*"))  # 같은 RUN_ID로 다시 실행하면 마지막 체크포인트부터 이어감
class DeptBatches(torch.utils.data.Sampler):  # 부처별로 섞고 64개씩 자른 뒤 묶음 순서를 섞음. 남는 것끼리는 섞인 묶음
    def __init__(self, dp, bs, seed):
        rng = random.Random(seed); g = {}
        for i, d in enumerate(dp): g.setdefault(d, []).append(i)
        self.b, rest = [], []
        for d in sorted(g, key=str):
            ix = g[d]; rng.shuffle(ix); k = len(ix) // bs * bs; self.b += [ix[j:j + bs] for j in range(0, k, bs)]; rest += ix[k:]
        rng.shuffle(rest); self.b += [rest[j:j + bs] for j in range(0, len(rest) // bs * bs, bs)]; rng.shuffle(self.b)
        self.same_dept_batches = len(self.b) - len(rest) // bs
    def __iter__(self): return iter(self.b)
    def __len__(self): return len(self.b)
class Trainer(SentenceTransformerTrainer):
    def get_batch_sampler(self, dataset, *a, **k):
        if DEPT_BATCH and dataset is ds: return DeptBatches(DP, HP["batch"], SEED)
        return super().get_batch_sampler(dataset, *a, **k)
DP = [r.get("dept") for r in prs] if PAIRS_FILE == "crop" else None
assert not DEPT_BATCH or DP, "DEPT_BATCH는 crop 전용"
res = Trainer(model=m, args=args, train_dataset=ds, loss=losses.MultipleNegativesRankingLoss(m, scale=float(os.environ.get("MNRL_SCALE", "20.0")))).train(resume_from_checkpoint=resume or None)
train_sec = time.time() - t0
m[0].auto_model.save_pretrained(str(out / "adapter"))

rows = [json.loads(l) for l in open("/data/encode_all_v1.jsonl", encoding="utf-8")]
if ENC_LIMIT: rows = rows[:ENC_LIMIT]
m.eval(); m.max_seq_length = ENC_LEN
t1 = time.time()
order = sorted(range(len(rows)), key=lambda i: len(rows[i]["text"]))  # 길이순 → 패딩 낭비 감소
v = np.zeros((len(rows), 4096), np.float32)
with torch.no_grad():
    for s in range(0, len(order), 512):
        idx = order[s:s + 512]
        v[idx] = m.encode([rows[i]["text"] for i in idx], batch_size=32, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        print(f"enc {min(s + 512, len(order))}/{len(order)}", flush=True)
enc_sec = time.time() - t1
np.savez(out / "qwen_chunk.npz", h=np.array([r["sha256"] for r in rows]), v=v)
meta = json.load(open("/data/" + ("corpus_meta_v2.json" if "v2" in CORPUS else "corpus_meta_v1.json"), encoding="utf-8"))  # 학습 입력 메타. 채점 입력은 encode_all_v1.jsonl(아래 rows)
man = {"run_id": RUN_ID, "method": "simcse_lora", "base_model": "Qwen/Qwen3-Embedding-8B", "base_model_dir": MODEL,
       "hyperparams": HP, "train_loss": res.training_loss, "seed": SEED, "train_max_seq_length": TRAIN_LEN, "train_corpus": CORPUS, "pairs_file": PAIRS_FILE, "pairs_file_sha256": sha("/data/" + PAIRS_FILE.removeprefix("crop+")) if PAIRS_FILE and PAIRS_FILE != "crop" else None, "pairs_in_file": npairs_file if PAIRS_FILE else None, "pairs_used": len(ds), "pairs_rule": (f"crop {HP['crop'][0]:.0%}~{HP['crop'][1]:.0%}/min{HP['crop'][2]}" if PAIRS_FILE == "crop" else ("crop+file 1:1, one_pair_per_program" if PAIRS_FILE.startswith("crop+") else "one_pair_per_program")) if PAIRS_FILE else "simcse_self", "max_steps": MAX_STEPS, "dept_batch": DEPT_BATCH, "mnrl_scale": float(os.environ.get("MNRL_SCALE", "20.0")), "limit_env": int(os.environ.get("LIMIT", "0")), "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1), "hypothesis_primary": HYPOTHESIS, "train_corpus_sha256": sha("/data/" + CORPUS), "encode_max_seq_length": ENC_LEN,
       "train_texts_used": len(corpus), "limit": LIMIT, "enc_limit": ENC_LIMIT, "train_seconds": round(train_sec, 1), "encode_seconds": round(enc_sec, 1),
       "prompt": None, "symmetric": True, "embedding_dim": int(v.shape[1]), "n": int(v.shape[0]), "dtype": "float32", "l2_normalized": True,
       "norm_check": [float(x) for x in np.linalg.norm(v[:3], axis=1)], "npz_sha256": sha(out / "qwen_chunk.npz"),
       "adapter_sha256": sha(out / "adapter" / "adapter_model.safetensors"),
       "inputs": meta, "image": "embed-ft:20261102-v1", "gpu": torch.cuda.get_device_name(0),
       "peak_gpu_mem_GB": round(torch.cuda.max_memory_allocated() / 2**30, 1),
       "reload_note": "어댑터 재로드는 peft.PeftModel.from_pretrained 사용 (transformers 5.3 load_adapter는 가중치 초기화 버그)"}
(out / "manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
print("RESULT", json.dumps({k: man[k] for k in ["run_id", "n", "train_loss", "train_seconds", "encode_seconds", "peak_gpu_mem_GB"]}, ensure_ascii=False))
