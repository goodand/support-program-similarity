import argparse,json,time,os
import numpy as np
import torch
from safetensors.torch import load_file,save_file
import rank_distill as rd
from evaluate_full_external import atomic_json,sha
parser=argparse.ArgumentParser()
parser.add_argument("--k",type=int,choices=[2,4,8],required=True)
k=parser.parse_args().k
source=rd.HERE/"main_rank_20261125"
out=source if k==4 else rd.HERE/f"main_rank_20261125_k{k}"
out.mkdir(exist_ok=True)
root=rd.HERE/"rank_pilot_20261125"
positions={2:[1,25],4:[1,3,9,25],8:[1,2,3,5,9,13,18,25]}[k]
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32=False
docs=json.loads((source/"docs.json").read_text("utf8"))
rd.texts=[r["text"] for r in docs]
rd.dec=rd.load_decider()
atomic_json(out/"status.json",dict(stage="waiting_for_teacher_vectors",k=k))
while not (source/"teacher.npy").exists() or (source/"teacher.npy").stat().st_size < len(docs)*4096*4+128:
    time.sleep(2)
x=torch.from_numpy(np.load(source/"teacher.npy")).cuda()
assert len(x)==len(docs) and torch.isfinite(x).all()
ids=np.random.default_rng(20261125).permutation(len(docs)).tolist()
def neighborhoods(pool):
    # Split before neighbor retrieval so validation documents cannot enter training.
    pool_tensor=torch.tensor(pool,device="cuda")
    v=x[pool_tensor]
    rows=[]
    for i in range(0,len(pool),128):
        sim=v[i:i+128]@v.T
        ix=torch.arange(len(sim),device="cuda")
        sim[ix,ix+i]=-torch.inf
        scores,ind=sim.topk(25,dim=1)
        scores,ind=scores.cpu().tolist(),ind.cpu().tolist()
        for j,ranked in enumerate(ind):
            for p in positions:
                rows.append(dict(q=pool[i+j],candidate=pool[ranked[p-1]],cosine=scores[j][p-1]))
    return rows
validation=neighborhoods(ids[:256])
train=neighborhoods(ids[256:])
atomic_json(out/"sampling.json",dict(seed=20261125,k=k,positions=positions,train=train,validation=validation))
atomic_json(out/"config.json",dict(k=k,train_queries=len(train)//k,validation_queries=256,lr=.0002,batch_queries=32,epochs=1,selection="minimum validation pairwise loss across quarter-epoch checkpoints including initialization",docs_sha256=sha(source/"docs.json"),teacher_vectors_sha256=sha(source/"teacher.npy"),code=sha(rd.HERE/"main_rank.py")))

initial=load_file(str(rd.MODEL/"readout.safetensors"))["weight"].float()
w=torch.nn.Parameter(initial[:2].cuda())
opt=torch.optim.AdamW([w],lr=.0002,weight_decay=0)
def checkpoint(path):
    weights=initial.clone()
    weights[:2]=w.detach().cpu()
    tmp=path.with_suffix(".tmp")
    save_file({"weight":weights},str(tmp))
    tmp.replace(path)
hv=None
best=float('inf')
if not (out/"resume.pt").exists():checkpoint(out/"best.safetensors")
batch=32*k
steps=(len(train)+batch-1)//batch
gates={max(1,round(steps*f/4)) for f in range(1,5)}
start=time.monotonic()
history=[]
resume_step=0
if (out/"resume.pt").exists():
    resume=torch.load(out/"resume.pt",weights_only=True,map_location="cuda")
    assert resume["k"]==k and resume["docs"]==sha(source/"docs.json")
    with torch.no_grad():w.copy_(resume["w"])
    opt.load_state_dict(resume["opt"])
    resume_step=resume["step"]
    history=resume["history"]
    best=min((r["validation"]["loss"] for r in history),default=float("inf"))
for step,i in enumerate(range(0,len(train),batch),1):
    if step<=resume_step:continue
    rows=train[i:i+batch]
    h=rd.features(rows)
    a,b=rd.comparisons(rows)
    scores=torch.nn.functional.linear(h,w)
    margin=scores[:,1]-scores[:,0]
    loss=torch.nn.functional.softplus(margin[b]-margin[a]).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    if step in gates and hv is None:
        hv=torch.cat([rd.features(validation[j:j+128]) for j in range(0,len(validation),128)])
        vp=rd.comparisons(validation)
        if not history:
            base=rd.assess(hv,initial[:2].cuda(),vp)
            best=base["loss"]
            history.append(dict(step=0,validation=base))
    if step in gates:
        val=rd.assess(hv,w,vp)
        history.append(dict(step=step,validation=val))
        if val["loss"]<best:
            checkpoint(out/"best.safetensors")
            best=val["loss"]
        atomic_json(out/"validation.json",history)
    if step==1 or step%10==0 or step in gates:
        checkpoint(out/"latest.safetensors")
        torch.save(dict(w=w.detach(),opt=opt.state_dict(),step=step,history=history,k=k,docs=sha(source/"docs.json")),out/"resume.tmp")
        (out/"resume.tmp").replace(out/"resume.pt")
        atomic_json(out/"status.json",dict(stage="training",step=step,steps=steps,queries=min(i+batch,len(train))//k,loss=loss.item(),seconds=time.monotonic()-start))
        print(step,steps,loss.item(),flush=True)
data,keys,_,queries=rd.load_panel()
old={q:set() for q in queries}
for a,b in data["eval_groups"].values():
    i,j=keys.index(a),keys.index(b)
    old[i].add(j);old[j].add(i)
bestw=load_file(str(out/"best.safetensors"))["weight"][:2].cuda()
records=[]
for q in queries:
    h=load_file(str(root/"external"/f"features_{q:03}.safetensors"))["hidden"].float().cuda()
    s=torch.nn.functional.linear(h,bestw)
    raw=(s[:,1]-s[:,0]).cpu().numpy()
    candidates=np.delete(np.arange(668),q)
    order=candidates[np.lexsort((candidates,-raw))].tolist()
    ranks=sorted(order.index(c)+1 for c in old[q])
    records.append(dict(q=q,ranks=ranks,ap=float(np.mean(np.arange(1,len(ranks)+1)/ranks))))
atomic_json(out/"external.json",dict(MAP=float(np.mean([r["ap"] for r in records])),worst=max(r["ranks"][-1] for r in records),all8=sum(r["ranks"][-1]<=8 for r in records),queries=records))
atomic_json(out/"status.json",dict(stage="complete",steps=steps,queries=len(train)//k,best_step=min(history,key=lambda z:z["validation"]["loss"])["step"]))





