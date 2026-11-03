"""Smoke test: Qwen3-Embedding-8B + LoRA, synthetic dummy pairs, MNRL 1 epoch, save adapter, reload, check dim/GPU."""
import json, time, torch
from datasets import Dataset
from peft import LoraConfig, TaskType
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, losses

MODEL = "/models/Qwen__Qwen3-Embedding-8B"
OUT = "/output/smoke_qwen_lora_" + time.strftime("%Y%m%d_%H%M%S")
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
assert torch.cuda.is_available(), "GPU not visible"

colors = ["red", "blue", "green", "gold", "gray"]
things = ["apple", "stone", "river", "cloud"]
pairs = [{"anchor": f"dummy {c} {t} number {i}", "positive": f"a {t} that is {c}, sample {i}"}
         for i, (c, t) in enumerate((c, t) for c in colors for t in things)]
ds = Dataset.from_list(pairs)

model = SentenceTransformer(MODEL, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
model.add_adapter(LoraConfig(task_type=TaskType.FEATURE_EXTRACTION, r=16, lora_alpha=32, lora_dropout=0.05,
                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
n_all = sum(p.numel() for p in model.parameters())
print("trainable", n_train, "of", n_all)

args = SentenceTransformerTrainingArguments(output_dir=OUT + "/ckpt", num_train_epochs=1, per_device_train_batch_size=4,
                                            learning_rate=1e-4, bf16=True, logging_steps=1, save_strategy="no",
                                            report_to="none", gradient_checkpointing=True)
trainer = SentenceTransformerTrainer(model=model, args=args, train_dataset=ds, loss=losses.MultipleNegativesRankingLoss(model))
res = trainer.train()
peak = round(torch.cuda.max_memory_allocated() / 1e9, 2)
print("train_loss", res.training_loss, "peak_gpu_mem_GB", peak)
probe = ["dummy red apple", "a stone that is gray"]
emb_trained = model.encode(probe, convert_to_tensor=True).float()
model[0].auto_model.save_pretrained(OUT + "/adapter")  # LoRA adapter only (base stays read-only in /models)
del model, trainer; torch.cuda.empty_cache()

m2 = SentenceTransformer(MODEL, device="cuda", model_kwargs={"torch_dtype": torch.bfloat16})
emb_base = m2.encode(probe, convert_to_tensor=True).float()
from peft import PeftModel
m2[0].auto_model = PeftModel.from_pretrained(m2[0].auto_model, OUT + "/adapter")  # transformers 5.3 load_adapter mis-maps keys
emb = m2.encode(probe, convert_to_tensor=True).float()
summary = {"out": OUT, "dim": int(emb.shape[1]), "device": str(emb.device), "train_loss": res.training_loss,
           "peak_gpu_mem_GB": peak, "trainable_params": n_train,
           "reload_vs_trained_maxdiff": float((emb - emb_trained).abs().max()),
           "reload_vs_base_maxdiff": float((emb - emb_base).abs().max()),
           "delta_cosine": float(torch.nn.functional.cosine_similarity((emb - emb_base).flatten(), (emb_trained - emb_base).flatten(), dim=0)),
           "gpu": torch.cuda.get_device_name(0)}
print("SMOKE_RESULT", json.dumps(summary))
