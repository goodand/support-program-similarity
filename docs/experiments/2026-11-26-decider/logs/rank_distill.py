"""Frozen Decider readout, within-query embedding teacher pairwise distillation."""
import json
import re
import sys
from itertools import combinations
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import save_file, load_file
from evaluate_full_external import HERE, MODEL, MODEL_SRC, INPUT, load_panel, request, atomic_json, sha
from frozen_head import forward_features

sys.path.insert(0, str(MODEL_SRC))
from pplx_decider import Decider
from pplx_decider.protocol import left_pad

def load_decider():
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
                                      max_length=2048, batch_tokens=4096)
    decider.backbone.requires_grad_(False).eval()
    return decider

def features(rows):
    chunks = []
    for i, row in enumerate(rows):
        chunk, = dec.encoder.plan(request(row["q"], row["candidate"], texts), 2048)
        chunks.append((i, chunk))
    result = [None] * len(rows)
    for batch in dec._batches(chunks):
        values = forward_features(dec.backbone, batch, left_pad, dec.encoder.pad_id, dec.device)
        for (i, _), value in zip(batch, values, strict=True):
            result[i] = value
    return torch.stack(result).float().cuda()

def comparisons(rows):
    groups = {}
    for i,r in enumerate(rows):
        groups.setdefault(r["q"], []).append(i)
    pairs = []
    for group in groups.values():
        for i,j in combinations(group, 2):
            gap = rows[i]["cosine"] - rows[j]["cosine"]
            if gap:
                pairs.append((i,j) if gap > 0 else (j,i))
    return torch.tensor(pairs,device="cuda").T

def assess(h,w,pairs):
    s = F.linear(h,w)
    margin = s[:,1] - s[:,0]
    gap = margin[pairs[0]] - margin[pairs[1]]
    return dict(correct=int((gap>0).sum()), total=gap.numel(), loss=F.softplus(-gap).mean().item())

def fit(name, rows, epochs, rate):
    h = features(rows)
    pair = comparisons(rows)
    w = torch.nn.Parameter(base.clone())
    before = assess(h,w,pair)
    opt = torch.optim.AdamW([w], lr=rate, weight_decay=0)
    for step in range(epochs):
        s = F.linear(h,w)
        m = s[:,1] - s[:,0]
        loss = F.softplus(m[pair[1]] - m[pair[0]]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if name=="smoke" and (step+1)%10==0 and assess(h,w,pair)["correct"]==pair.shape[1]:
            break
    checkpoint = initial.clone()
    checkpoint[:2] = w.detach().cpu()
    path = out / (name+".safetensors")
    save_file({"weight":checkpoint},str(path))
    restored = load_file(str(path))["weight"][:2].cuda()
    assert torch.equal(restored,w.detach())
    after=assess(h,restored,pair)
    report=dict(before=before,after=after,steps=step+1,lr=rate,checkpoint_sha256=sha(path))
    atomic_json(out/(name+".json"),report)
    print(name,report,flush=True)
    return restored, report

def make_rows(queries, allowed, x):
    rows=[]
    for q in queries:
        ids=np.array(sorted(allowed-{q}))
        ranked=ids[np.argsort(-(x[ids]@x[q]),kind="stable")]
        for c in ranked[[0,2,8,24]]:
            rows.append(dict(q=int(q),candidate=int(c),cosine=float(x[q]@x[c])))
    return rows

def teach_vectors():
    import hashlib
    parent=INPUT.parent.parent
    manifest=json.loads((parent/"pool668_embedding_only_20261123_harness/manifest_harness.json").read_text("utf8"))
    vectors={}
    for p in [Path(manifest["sources"]["post"][0]["path"]),parent/"emb_pool035/qwen_cropB3batch128_mem5_s42_20261106_night3__new2_harness.npz"]:
        with np.load(p) as z:
            vectors.update(zip(z["h"].tolist(),z["v"]))
    x=np.stack([vectors[hashlib.sha256(t.encode()).hexdigest()] for t in texts]).astype("float32")
    return x/np.linalg.norm(x,axis=1)[:,None]

if __name__=="__main__":
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32=False
    out=Path(sys.argv[1])
    out.mkdir(parents=True,exist_ok=False)
    data,keys,texts,external=load_panel()
    def title(t): return re.sub(r"\s+","",t.split(" / ")[0].split(":",1)[1])
    blocked={title(texts[q]) for q in external}
    allowed={i for i,t in enumerate(texts) if title(t) not in blocked}
    smoke=json.loads((HERE/"teacher32_embedding.json").read_text())
    smoke_ids={r[k] for r in smoke for k in ("q","candidate")}
    assert smoke_ids<=allowed and len(smoke)==32
    x=teach_vectors()
    assert all(abs(r["cosine"]-float(x[r["q"]]@x[r["candidate"]]))<1e-5 for r in smoke)
    atomic_json(out/"manifest.json",dict(seed=20261125,panel=sha(INPUT),smoke=sha(HERE/"teacher32_embedding.json"),base_head=sha(MODEL/"readout.safetensors"),code=sha(Path(__file__)),allowed=len(allowed),external=external))
    dec=load_decider()
    initial=load_file(str(MODEL/"readout.safetensors"))["weight"].float()
    base=initial[:2].cuda()
    _,report=fit("smoke",smoke,400,0.001)
    if report["after"]["correct"]!=report["after"]["total"] or report["after"]["loss"]>=report["before"]["loss"]:
        sys.exit("smoke gate failed")
    # Keep the held-out teacher queries out of every training role.
    pool=allowed-smoke_ids
    rng=np.random.default_rng(20261125)
    chosen=rng.choice(sorted(pool),80,replace=False).tolist()
    train=make_rows(chosen[:64],pool-set(chosen[64:]),x)
    validation=make_rows(chosen[64:],pool-set(chosen[:64]),x)
    atomic_json(out/"sample.json",dict(train=train,validation=validation))
    w,_=fit("pilot",train,60,0.0002)
    hv=features(validation)
    p=comparisons(validation)
    held=dict(base=assess(hv,base,p),pilot=assess(hv,w,p))
    atomic_json(out/"heldout.json",held)
    print("heldout",held,flush=True)
