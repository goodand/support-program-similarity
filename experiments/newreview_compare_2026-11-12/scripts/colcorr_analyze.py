"""열 상관 분석(학습 후 B3+배치128 seed 42 임베딩). (1) 열별 유사도 행렬 간 Spearman(공고 쌍 단위),
(2) 열 하나만 / 9칸에서 열 하나 뺀 입력의 신규검토 지표(eval_v3 rank, 질의 35개). 빈 값 공고는 해당 열 계산에서 제외.
usage: <EMBED_ENV_PYTHON> -I -X utf8 runs/044/colcorr_analyze_harness.py
출력: runs/044/colcorr_harness/colcorr_result_harness.json"""
import os
import json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
sys.stdout.reconfigure(encoding='utf-8')
KIM = Path(__file__).resolve().parents[2]; O = KIM / 'runs/044/colcorr_harness'
EVF = KIM / 'runs/044/pool668_newreview_harness/eval_set_pool668.json'
src = (KIM / 'eval_v3.py').read_text(encoding='utf-8'); src = src.replace("EVAL = HERE / 'runs' / '009' / 'eval_set_v2.json'", f"EVAL = Path({str(EVF)!r})")
sys.argv = ['eval_v3.py', '044/pool668_newreview_harness', '30']; g = {'__file__': str(KIM / 'eval_v3.py'), '__name__': 'core'}; exec(src[:src.index('cache = {}')], g)
R, ix, ev, rank, N = g['R'], g['ix'], g['ev'], g['rank'], g['N']
T = defaultdict(set)
for gr, a, b in ev['pairs']: T[gr] |= {ix[a], ix[b]}
QS = sorted((gr, q) for gr, m in T.items() for q in m)
idx = json.load(open(O / 'index.json'))
with np.load(O / 'qwen_colcorr_b3b128.npz') as z: e = dict(zip(z['h'].tolist(), z['v']))
def mat(k):
    hs = idx[k]; X = np.full((N, 4096), np.nan, np.float32)
    for i, h in enumerate(hs):
        if h: X[i] = e[h]
    return X @ X.T, np.array([h is not None for h in hs]), len(set(h for h in hs if h))
iu = np.triu_indices(N, 1); out = {'metrics': {}, 'n_unique_values': {}, 'spearman': {}}
cols = [k for k in idx if k.startswith('col:')]; Ms = {}
for k in idx:
    S, ok, nu = mat(k); last, miss = [], 0
    for gr, q in QS:
        if not ok[q]: miss += 1; last.append(None); continue
        sc = S[q].astype(float).copy(); sc[q] = np.nan; sc[~ok] = -np.inf
        last.append(max(rank(sc, q, o) for o in T[gr] if o != q))
    L = [x for x in last if x is not None]
    out['metrics'][k] = {'worst': max(L), 'mean': float(np.mean(L)), 'median': float(np.median(L)), 'query_missing': miss, 'last': last}
    out['n_unique_values'][k] = nu
    if k.startswith('col:'): Ms[k] = (S, ok)
    print(k, 'worst', max(L), 'mean', round(float(np.mean(L)), 2), '고유값', nu, flush=True)
for a in cols:
    for b in cols:
        if a >= b: continue
        Sa, oa = Ms[a]; Sb, ob = Ms[b]; m = (oa[iu[0]] & oa[iu[1]] & ob[iu[0]] & ob[iu[1]])
        out['spearman'][f'{a}|{b}'] = float(spearmanr(Sa[iu][m], Sb[iu][m]).statistic)
json.dump(out, open(O / 'colcorr_result_harness.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
names = [c[4:] for c in cols]; print('\t' + '\t'.join(n[:4] for n in names))
for a in cols:
    print(a[4:8], '\t'.join(('1.00' if a == b else f"{out['spearman'].get(f'{a}|{b}', out['spearman'].get(f'{b}|{a}')):.2f}") for b in cols))
