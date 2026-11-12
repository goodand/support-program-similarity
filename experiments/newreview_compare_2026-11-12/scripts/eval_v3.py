"""Retrieval evaluation on a large pool without the N x N matrix (eval_v2 recipe, exact ranks).

S = (W + W^T)/2, W = 0.5*rownorm(VN_sym) + 0.5*rownorm(Qwen cos), as eval_v2. Row min/max for every row come from
knn_build.py (runs/<run>/knn/knn_chunk_K<K>.npz 'mm'); only the query rows are recomputed here, so S[q, :] is exact.
VN algebra is copied from knn_build.py (same as eval_pipeline.vn_parts, asserted equal to exp_M05_VN.vn_field there).
Eval pairs: runs/009/eval_set_v2.json (both directions). Alias: an eval element and its single same-budget 세부/내역
twin (세부 with exactly one 내역, budget equal ±1) — the query never counts its own twin, and a positive is found when it or
its twin is retrieved (사용자 2026-10-30 rule, applied only to eval elements).
Usage: <EMBED_ENV_PYTHON> -I -X utf8 eval_v3.py <run> [K] [qalias]
qalias: the query's twin is searched too and each candidate keeps the higher score (사용자 2026-11-03 '일단 해봐').
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs' / sys.argv[1]
K = int(sys.argv[2]) if len(sys.argv) > 2 else 30
QALIAS = 'qalias' in sys.argv[3:]  # also search with the query's twin; per candidate, keep the higher score
EVAL = HERE / 'runs' / '009' / 'eval_set_v2.json'
VN = dict(bv=1.0, bs=0.75, k1=1.5)

R = [json.loads(l) for l in open(RUN / 'chunks.jsonl', encoding='utf-8')]
key = lambda r: r['type'] + '|' + json.dumps(r['id'], ensure_ascii=False, sort_keys=True)  # noqa: E731
keys = [key(r) for r in R]; ix = {k: i for i, k in enumerate(keys)}; N = len(keys)
knn = np.load(RUN / 'knn' / f'knn_chunk_K{K}.npz')
assert knn['keys'].tolist() == keys, 'knn keys differ from chunks'
mm = knn['mm']
with np.load(RUN / 'emb_cache' / 'qwen_chunk.npz') as z:
    emb = dict(zip(z['h'].tolist(), z['v']))
E = np.stack([emb[r['sha256']] for r in R]).astype(np.float32); del emb
TOK = json.loads((RUN / 'kiwi_tokens.json').read_text(encoding='utf-8'))
toks = [TOK[hashlib.sha256(r['text'].encode('utf-8')).hexdigest()] for r in R]
vocab, rows, cols, vals = {}, [], [], []
for i, d in enumerate(toks):
    for w, n in Counter(d).items():
        rows.append(i); cols.append(vocab.setdefault(w, len(vocab))); vals.append(n)
tf = sp.csr_matrix((np.array(vals, np.float64), (rows, cols)), shape=(N, len(vocab)))
V = np.array([len(d) / max(1, len(set(d))) for d in toks]); S_ = np.array([len(set(d)) for d in toks], float)
aV, aS = max(1e-9, V.mean()), max(1e-9, S_.mean())
df = np.asarray((tf > 0).sum(0)).ravel(); idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
c = tf.tocoo()
x = c.data / ((1 - VN['bv']) + VN['bv'] * V[c.row] / aV) / ((1 - VN['bs']) + VN['bs'] * S_[c.row] / aS)
sat = sp.csr_matrix((x / (VN['k1'] + x), (c.row, c.col)), shape=tf.shape)
qw = tf.multiply(idf[None, :]).tocsr()
satT, qwT = sat.T.tocsr(), qw.T.tocsr()


def nz(v, lo, hi):
    return (v - lo) / np.where(hi - lo > 0, hi - lo, 1)


def rows_of(q):
    """exact VN, Qwen and combined S rows for query q (diagonal = nan)"""
    fwd = np.asarray((qw[q] @ satT).todense()).ravel()   # q->c
    bwd = np.asarray((sat[q] @ qwT).todense()).ravel()   # c->q
    vn = 0.5 * (fwd + bwd)                               # symmetric: VN[q,c] = VN[c,q]
    qc = E @ E[q]
    vn[q] = qc[q] = np.nan
    Wq = 0.5 * nz(vn, mm[q, 0], mm[q, 1]) + 0.5 * nz(qc, mm[q, 2], mm[q, 3])      # W[q, c]
    Wc = 0.5 * nz(vn, mm[:, 0], mm[:, 1]) + 0.5 * nz(qc, mm[:, 2], mm[:, 3])      # W[c, q]
    return {'VN': vn, 'Qwen': qc, 'VN+Qwen': 0.5 * (Wq + Wc)}


# twin alias (single 내역 = its 세부, same budget) for eval elements only
by_sid = defaultdict(list)
for i, r in enumerate(R):
    if r['type'] in ('세부사업', '내역사업') and 'id' in r and '사업_id' in r['id']:
        by_sid[(r.get('source', '')[:14], r['id']['사업_id'])].append(i)
ev = json.loads(EVAL.read_text(encoding='utf-8'))
eval_ix = {ix[r['chunk']['key']] for r in ev['rows'] if r.get('chunk')}
alias = {}
for grp in by_sid.values():
    top = [i for i in grp if R[i]['type'] == '세부사업']; det = [i for i in grp if R[i]['type'] == '내역사업']
    if len(top) == 1 and len(det) == 1 and (top[0] in eval_ix or det[0] in eval_ix):
        bt, bd = R[top[0]].get('budget'), R[det[0]].get('budget')
        if bt is not None and bd is not None and abs(float(bt) - float(bd)) <= 1:
            alias[top[0]], alias[det[0]] = det[0], top[0]


def rank(s, q, p):
    s = s.copy(); s[q] = -np.inf
    s = np.nan_to_num(s, nan=-np.inf)
    if q in alias: s[alias[q]] = -np.inf
    cand = [p] + ([alias[p]] if p in alias else [])
    v = max(s[c] for c in cand)
    return 1 + int((s > v).sum()) + ((s == v).sum() - 1) / 2


cache = {}
out = []
for g, a, b in ev['pairs']:
    for q, p in ((ix[a], ix[b]), (ix[b], ix[a])):
        if q not in cache:
            cache[q] = rows_of(q)
            if QALIAS and q in alias:
                t = rows_of(alias[q])
                cache[q] = {n: np.fmax(v, t[n]) for n, v in cache[q].items()}
        for name, s in cache[q].items():
            out.append({'sim': name, 'group': g, 'query': R[q]['name'], 'positive': R[p]['name'], 'rank': rank(s, q, p)})
summ = {}
for name in ('VN', 'Qwen', 'VN+Qwen'):
    rk = np.array([o['rank'] for o in out if o['sim'] == name])
    summ[name] = {'n': int(len(rk)), 'MRR': float(np.mean(1 / rk)), 'R@1': float(np.mean(rk <= 1)), 'R@10': float(np.mean(rk <= 10)),
                  'R@30': float(np.mean(rk <= 30)), 'R@100': float(np.mean(rk <= 100)), 'median_rank': float(np.median(rk))}
res = {'run': RUN.name, 'qalias': QALIAS, 'alias_pairs': sorted([R[a]['type'] + ':' + R[a]['name'] for a in alias]), 'N': N, 'pairs': len(ev['pairs']), 'aliases': len(alias) // 2, 'summary': summ, 'rows': out,
       'inputs': {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (RUN / 'chunks.jsonl', EVAL, RUN / 'kiwi_tokens.json', Path(__file__))}}
(RUN / ('eval_v3_qalias.json' if QALIAS else 'eval_v3.json')).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({'N': N, 'pairs': len(ev['pairs']), 'aliases': len(alias) // 2, 'summary': summ}, ensure_ascii=False, indent=1))
