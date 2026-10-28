"""kNN graph of the 2026 central pool for Leiden clustering (CPU/RAM only; uses cached Qwen embeddings, no GPU).

Similarity = the budget-clustering recipe (cluster_grid chosen config, minus the reranker which would need the GPU):
  W = 0.5*rownorm(VN_sym) + 0.5*rownorm(Qwen cosine),  S = (W + W^T)/2
rownorm = row-wise min-max over the other items (diagonal excluded), as cluster_grid.rownorm. VN = exp_M05_VN two-stage
BM25 (bv=1, bs=0.75, k1=1.5) on Kiwi tokens, made symmetric by 0.5*(q->c + c->q).
The full 36,889^2 matrix is never stored: rows are processed in blocks (threads), each row keeps its top-K neighbours,
then S is computed exactly for every kept pair (both directions' row min/max are known).
Usage: <DEV_EVAL_ROOT>/_embedding_v3_env/Scripts/python.exe -I -X utf8 knn_build.py <run> <chunk_type> [K]
  -> runs/<run>/knn/knn_<type>_K<K>.npz  (pairs i<j, S, plus per-row min/max)
"""
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs' / sys.argv[1]
TYPE = sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 30
SIM = sys.argv[4] if len(sys.argv) > 4 else 'vn+qwen'  # 'vn' = lexical only (no embeddings needed; CPU while the GPU embeds)
# 'vnfwd' = lexical only, one direction (q->c) before the row-norm + (W+W^T)/2 symmetrization (2026-10-28: VN_sym hurt retrieval)
OUTP = RUN / 'knn' / (f'knn_{TYPE}_K{K}.npz' if SIM == 'vn+qwen' else f'knn_{TYPE}_K{K}_{SIM}.npz')
OUTP.parent.mkdir(exist_ok=True)
VN = dict(bv=1.0, bs=0.75, k1=1.5)
BLOCK = 1024

T = json.loads((RUN / 'texts.json').read_text(encoding='utf-8'))
TOK = json.loads((RUN / 'kiwi_tokens.json').read_text(encoding='utf-8'))
keys = T['keys']; N = len(keys)
sha = lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest()  # noqa: E731
texts = [T['texts'][TYPE][k] for k in keys]
hs = [sha(x) for x in texts]

# Qwen embeddings from the shared text-sha cache (all qwen_*.npz)
have = {}
if SIM in ('vn', 'vnfwd'):
    E = np.zeros((N, 1), np.float32)  # unused; W = rownorm(VN_sym) only
else:
    for f in sorted((RUN / 'emb_cache').glob('qwen_*.npz')):
        if '.tmp' in f.name:
            continue
        z = np.load(f); have.update(zip(z['h'].tolist(), z['v']))
    miss = sum(h not in have for h in hs)
    assert miss == 0, f'{miss} texts of {TYPE} have no embedding yet'
    E = np.stack([have[h] for h in hs]).astype(np.float32)
    del have
QW = 0.0 if SIM in ('vn', 'vnfwd') else 0.5
FWD_ONLY = SIM == 'vnfwd'  # weight of the Qwen part

# VN parts (same algebra as eval_pipeline.vn_parts, which is asserted equal to exp_M05_VN.vn_field)
toks = [TOK[h] for h in hs]
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


def raw_rows(a, b):
    """VN_sym and Qwen rows a:b (dense, diagonal set to nan)"""
    fwd = (qw[a:b] @ satT).toarray()
    bwd = (sat[a:b] @ qwT).toarray()
    vn = fwd if FWD_ONLY else 0.5 * (fwd + bwd)
    qc = E[a:b] @ E.T
    for r in range(b - a):
        vn[r, a + r] = np.nan; qc[r, a + r] = np.nan
    return vn, qc


def block(a):
    b = min(N, a + BLOCK)
    vn, qc = raw_rows(a, b)
    vlo, vhi = np.nanmin(vn, 1), np.nanmax(vn, 1); qlo, qhi = np.nanmin(qc, 1), np.nanmax(qc, 1)
    W = 0.5 * (vn - vlo[:, None]) / np.where(vhi - vlo > 0, vhi - vlo, 1)[:, None] \
        + 0.5 * (qc - qlo[:, None]) / np.where(qhi - qlo > 0, qhi - qlo, 1)[:, None]
    W = np.nan_to_num(W, nan=-np.inf)
    top = np.argpartition(-W, K, axis=1)[:, :K]
    return a, top, vlo, vhi, qlo, qhi


nbr = np.zeros((N, K), np.int64)
mm = np.zeros((N, 4))
with ThreadPoolExecutor(16) as ex:
    for n, (a, top, vlo, vhi, qlo, qhi) in enumerate(ex.map(block, range(0, N, BLOCK)), 1):
        b = a + len(top)
        nbr[a:b] = top; mm[a:b] = np.stack([vlo, vhi, qlo, qhi], 1)
        if n % 8 == 0:
            print(f'{b}/{N}', flush=True)

# exact S for every kept pair
pairs = np.unique(np.sort(np.stack([np.repeat(np.arange(N), K), nbr.ravel()], 1), axis=1), axis=0)
pairs = pairs[pairs[:, 0] != pairs[:, 1]]
i, j = pairs[:, 0], pairs[:, 1]
vf = np.asarray(qw[i].multiply(sat[j]).sum(1)).ravel()   # i->j
vb = np.asarray(sat[i].multiply(qw[j]).sum(1)).ravel()   # j->i
vij = vf if FWD_ONLY else 0.5 * (vf + vb)
vji = vb if FWD_ONLY else vij
qij = np.einsum('nd,nd->n', E[i], E[j])
nv = lambda v, r: (v - mm[r, 0]) / np.where(mm[r, 1] - mm[r, 0] > 0, mm[r, 1] - mm[r, 0], 1)  # noqa: E731
nq = lambda v, r: (v - mm[r, 2]) / np.where(mm[r, 3] - mm[r, 2] > 0, mm[r, 3] - mm[r, 2], 1)  # noqa: E731
Wij = (1 - QW) * nv(vij, i) + QW * nq(qij, i)
Wji = (1 - QW) * nv(vji, j) + QW * nq(qij, j)
S = 0.5 * (Wij + Wji)
tmp = OUTP.with_name(OUTP.stem + '.tmp.npz')
np.savez(tmp, i=i, j=j, S=S, nbr=nbr, mm=mm, keys=np.array(keys))
tmp.replace(OUTP)
print(json.dumps({'type': TYPE, 'K': K, 'N': N, 'pairs': int(len(S)),
                  'texts_sha256': hashlib.sha256((RUN / 'texts.json').read_bytes()).hexdigest(),
                  'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}, ensure_ascii=False))
