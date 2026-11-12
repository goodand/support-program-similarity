"""5스텝 기준 설정(B3+배치128, MAX_STEPS=5)의 seed 42·43·45·46 재현성. 668 풀, 질의 35개, 단독(Qwen)·결합(VN+Qwen).
채점은 score_pool668_newreview_harness.py와 같다(eval_v3.py 앞부분 exec, 임베딩만 교체, Qwen 행 min/max 재계산).
usage: <EMBED_ENV_PYTHON> -I -X utf8 runs/044/score_seed_rep_harness.py
출력: runs/044/seed_rep_harness/seed_rep_scores_harness.json"""
import json, sys, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
KIM = Path(__file__).resolve().parents[2]; E = KIM / 'runs/044/seed_rep_harness'
EVF = KIM / 'runs/044/pool668_newreview_harness/eval_set_pool668.json'
src = (KIM / 'eval_v3.py').read_text(encoding='utf-8'); old = "EVAL = HERE / 'runs' / '009' / 'eval_set_v2.json'"; assert src.count(old) == 1
src = src.replace(old, f"EVAL = Path({str(EVF)!r})"); sys.argv = ['eval_v3.py', '044/pool668_newreview_harness', '30']
g = {'__file__': str(KIM / 'eval_v3.py'), '__name__': 'core'}; exec(src[:src.index('cache = {}')], g)
R, ix, ev, rank, N = g['R'], g['ix'], g['ev'], g['rank'], g['N']
T = defaultdict(set)
for gr, a, b in ev['pairs']: T[gr] |= {ix[a], ix[b]}
QS = sorted((gr, q) for gr, m in T.items() for q in m); assert len(QS) == 35
mm0 = g['mm'].copy()
def rowmm(X):
    S = X @ X.T; np.fill_diagonal(S, np.inf); lo = S.min(1); np.fill_diagonal(S, -np.inf); hi = S.max(1); return lo, hi
out = {'queries': 35, 'runs': {}}
for s in (42, 43, 45, 46):
    f = E / f'qwen668_s{s}.npz'
    with np.load(f) as z: e = dict(zip(z['h'].tolist(), z['v']))
    X = np.stack([e[r['sha256']] for r in R]).astype(np.float32)
    g['E'] = X; mm = mm0.copy(); mm[:, 2], mm[:, 3] = rowmm(X); g['mm'] = mm
    res = {'Qwen': [], 'VN+Qwen': []}
    for gr, q in QS:
        rw = g['rows_of'](q)
        for k in res:
            sc = np.array(rw[k], float); sc[q] = np.nan
            res[k].append(max(rank(sc, q, o) for o in T[gr] if o != q))
    out['runs'][f'seed {s}'] = {k: {'worst': max(v), 'mean': float(np.mean(v)), 'last': v} for k, v in res.items()}
    out['runs'][f'seed {s}']['npz_sha256'] = hashlib.sha256(f.read_bytes()).hexdigest()
    print(f'seed {s}', {k: (max(v), round(float(np.mean(v)), 2)) for k, v in res.items()}, flush=True)
json.dump(out, open(E / 'seed_rep_scores_harness.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
