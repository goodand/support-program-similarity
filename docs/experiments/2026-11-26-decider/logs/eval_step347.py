from evaluate_full_external import *
from safetensors.torch import load_file
import numpy as np, torch
root=HERE/"main_rank_20261125"
cache=HERE/"rank_pilot_20261125"/"external"
data,keys,texts,qs=load_panel()
truth={q:set() for q in qs}
for a,b in data["eval_groups"].values():
    i,j=keys.index(a),keys.index(b)
    truth[i].add(j)
    truth[j].add(i)
expanded={q:s.copy() for q,s in truth.items()}
titles=["".join(t.split(" / ")[0].split(":",1)[1].split()) for t in texts]
for row in json.loads((HERE/"expert_high9.json").read_text("utf8")):
    q,=[i for i in qs if titles[i]=="".join(row["query"].split())]
    c,=[i for i,t in enumerate(titles) if t=="".join(row["candidate"].split())]
    expanded[q].add(c)
assert sum(map(len,expanded.values()))==69
ck=root/"best.safetensors"
assert min(json.loads((root/"validation.json").read_text()),key=lambda x:x["validation"]["loss"])["step"]==347
torch.backends.cuda.matmul.allow_tf32=False
summary=[]
records=[]
for name,path in [('base',MODEL/'readout.safetensors'),('step347',ck)]:
    w=load_file(str(path))['weight'][:2].float().cuda()
    for q in qs:
        h=load_file(str(cache/f'features_{q:03}.safetensors'))['hidden'].float().cuda()
        s=h@w.T
        v=(s[:,1]-s[:,0]).cpu().numpy()
        assert v.shape==(667,) and np.isfinite(v).all()
        ids=np.delete(np.arange(668),q)
        order=ids[np.lexsort((ids,-v))].tolist()
        for label,g in [('original',truth),('plus9',expanded)]:
            ranks=sorted(order.index(i)+1 for i in g[q])
            records.append(dict(model=name,labels=label,q=q,ranks=ranks,ap=np.mean(np.arange(1,len(ranks)+1)/ranks)))
    for label in ['original','plus9']:
        rows=[r for r in records if r['model']==name and r['labels']==label]
        summary.append(dict(model=name,labels=label,map=np.mean([r['ap'] for r in rows]),worst=max(r['ranks'][-1] for r in rows),all8=sum(r['ranks'][-1]<=8 for r in rows)))
ref=json.loads((HERE/'expert_expanded9_metrics.json').read_text('utf8'))['summary']
for r,b in zip(summary[:2],[r for r in ref if r['model']=='base']):
    assert abs(r['map']-b['map'])<1e-9 and r['worst']==b['worst']
summary+=[r for r in ref if r['model']=='embedding']
atomic_json(root/'external_step347.json',dict(step=347,sha256=sha(ck),summary=summary,queries=records))
print(summary)
