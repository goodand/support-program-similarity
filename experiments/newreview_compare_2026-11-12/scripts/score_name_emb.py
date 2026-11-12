"""공고명만 임베딩(학습 전 base / 학습 후 B3+배치128 seed 42)의 신규검토 지표. 순위는 eval_v3.py의 rank()를 그대로 쓴다(score_pool668_newreview_harness.py와 같은 exec 방식).
usage: <EMBED_ENV_PYTHON> -I -X utf8 runs/044/score_name_emb_harness.py
출력: runs/044/name_enc_harness/name_emb_scores_harness.json"""
import os
import json, sys, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
KIM = Path(__file__).resolve().parents[2]; E = KIM / 'runs/044/name_enc_harness'
EVF = KIM / 'runs/044/pool668_newreview_harness/eval_set_pool668.json'
src = (KIM / 'eval_v3.py').read_text(encoding='utf-8'); old = "EVAL = HERE / 'runs' / '009' / 'eval_set_v2.json'"; assert src.count(old) == 1
src = src.replace(old, f"EVAL = Path({str(EVF)!r})"); sys.argv = ['eval_v3.py', '044/pool668_newreview_harness', '30']
g = {'__file__': str(KIM / 'eval_v3.py'), '__name__': 'core'}; exec(src[:src.index('cache = {}')], g)
R, ix, ev, rank, N = g['R'], g['ix'], g['ev'], g['rank'], g['N']
C = [json.loads(l) for l in open(KIM / 'runs/044/pool668_newreview_harness/chunks.jsonl', encoding='utf-8')]
assert len(C) == N
T = defaultdict(set)
for gr, a, b in ev['pairs']: T[gr] |= {ix[a], ix[b]}
QS = sorted((gr, q) for gr, m in T.items() for q in m); assert len(QS) == 35 and len(T) == 14
out = {'N': N, 'queries': 35, 'runs': {}}
for lab, f in [('학습 전 (base)', 'qwen_name_base.npz'), ('학습 후 B3+배치128 seed 42', 'qwen_name_b3b128.npz')]:
    with np.load(E / f) as z: emb = dict(zip(z['h'].tolist(), z['v']))
    X = np.stack([emb[hashlib.sha256(c['name'].encode()).hexdigest()] for c in C]).astype(np.float32)
    S = X @ X.T; last = []
    for gr, q in QS:
        sc = S[q].astype(float).copy(); sc[q] = np.nan
        last.append(max(rank(sc, q, o) for o in T[gr] if o != q))
    L = np.array(last, float)
    out['runs'][lab] = {'worst': float(L.max()), 'mean': float(L.mean()), 'median': float(np.median(L)), 'last': last,
                        'npz_sha256': hashlib.sha256((E / f).read_bytes()).hexdigest()}
    print(lab, {k: v for k, v in out['runs'][lab].items() if k not in ('last', 'npz_sha256')})
json.dump(out, open(E / 'name_emb_scores_harness.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
