"""Full-panel retrieval for a rank-distilled readout; cache reusable hidden states."""
import sys
from pathlib import Path
import json
import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file
import rank_distill as rd
from evaluate_full_external import atomic_json, sha

def main(out):
    data,keys,rd.texts,queries=rd.load_panel()
    rd.dec=rd.load_decider()
    rd.dec.backbone.eval()
    weights=[load_file(str(p))["weight"][:2].float().cuda() for p in (rd.MODEL/"readout.safetensors",out/"pilot.safetensors")]
    cache=out/"external"
    cache.mkdir(exist_ok=True)
    old={q:set() for q in queries}
    for a,b in data["eval_groups"].values():
        i,j=keys.index(a),keys.index(b)
        old[i].add(j)
        old[j].add(i)
    extra={q:s.copy() for q,s in old.items()}
    labels=json.loads((rd.HERE/"expert_high9.json").read_text("utf8"))
    clean=lambda s: "".join(s.split())
    titles=[clean(t.split(" / ")[0].split(":",1)[1]) for t in rd.texts]
    for row in labels:
        q,=[i for i in queries if titles[i]==clean(row["query"])]
        c,=[j for j,t in enumerate(titles) if t==clean(row["candidate"])]
        extra[q].add(c)
    records=[]
    for i,q in enumerate(queries):
        ids=np.array([j for j in range(668) if j!=q])
        path=cache/f"features_{q:03}.safetensors"
        if path.exists():
            h=load_file(str(path))["hidden"].cuda().float()
        else:
            h=rd.features([dict(q=q,candidate=j) for j in ids])
            from safetensors.torch import save_file
            tmp=path.with_suffix(".tmp")
            save_file({"hidden":h.bfloat16().cpu()},str(tmp))
            tmp.replace(path)
        for model,w in zip(("base","pilot"),weights):
            logits=F.linear(h,w)
            scores=(logits[:,1]-logits[:,0]).detach().cpu().numpy()
            assert np.isfinite(scores).all()
            ranked=ids[np.lexsort((ids,-scores))].tolist()
            for version,truth in (("prior",old),("plus9",extra)):
                ranks=sorted(ranked.index(c)+1 for c in truth[q])
                records.append(dict(q=q,model=model,version=version,ranks=ranks,ap=float(np.mean(np.arange(1,len(ranks)+1)/ranks))))
        atomic_json(cache/"progress.json",{"done":i+1,"total":35})
    summary=[]
    for m in ("base","pilot"):
        for v in ("prior","plus9"):
            subset=[r for r in records if r["model"]==m and r["version"]==v]
            summary.append(dict(model=m,version=v,MAP=float(np.mean([r["ap"] for r in subset])),worst=max(r["ranks"][-1] for r in subset),all8=sum(r["ranks"][-1]<=8 for r in subset)))
    atomic_json(cache/"metrics.json",dict(checkpoint=sha(out/"pilot.safetensors"),summary=summary,rows=records))
    print(summary,flush=True)

if __name__=="__main__":
    torch.backends.cuda.matmul.allow_tf32=False
    main(Path(sys.argv[1]))
