"""Leiden clustering of the 2026 central pool on a kNN graph (CPU only) and PU-safe evaluation on the 3 exact-match pairs.

Graph: from knn_build.py output, each item links to its top-k neighbours by S (undirected, one edge per pair, weight S
clipped to >= 1e-6) — same edge rule as cluster_grid.run_unit. Partition: leidenalg RBConfigurationVertexPartition,
grid k x resolution, 10 seeds (cluster_grid convention). Pre-declared primary = k=5, resolution=5.0 (the budget-clustering
choice). Metrics per config: pairs together (out of 3) per seed, cost = members of the query's cluster minus 1 (what a
reviewer reads), number of clusters, largest cluster. Items outside a gold pair are unlabeled (never negatives).
Usage: <DEV_EVAL_ROOT>/jh_dev/program_mapping_nlp_v1/.venv/Scripts/python.exe -X utf8 leiden_eval.py <run> <chunk_type> [K]
"""
import hashlib
import json
import sys
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs' / sys.argv[1]
TYPE = sys.argv[2]
K = int(sys.argv[3]) if len(sys.argv) > 3 else 30
SIM = sys.argv[4] if len(sys.argv) > 4 else 'vn+qwen'
SRC = RUN / 'knn' / (f'knn_{TYPE}_K{K}.npz' if SIM == 'vn+qwen' else f'knn_{TYPE}_K{K}_{SIM}.npz')
z = np.load(SRC)
keys = z['keys'].tolist(); N = len(keys); kix = {k: i for i, k in enumerate(keys)}
pS = {(int(a), int(b)): float(s) for a, b, s in zip(z['i'], z['j'], z['S'])}
nbr = z['nbr']
T = json.loads((RUN / 'texts.json').read_text(encoding='utf-8'))
pairs = [(g, ks[0], ks[1]) for g, ks in T['eval_groups'].items()]
KS, RES, SEEDS = [5, 10, 20], [1.0, 2.0, 5.0, 10.0, 20.0, 50.0], 10


def graph(k):
    ew = {}
    for i in range(N):
        cand = sorted(((pS[(min(i, int(j)), max(i, int(j)))], int(j)) for j in nbr[i] if int(j) != i), reverse=True)[:k]
        for s, j in cand:
            e = (min(i, j), max(i, j))
            ew[e] = max(ew.get(e, 0.0), max(s, 1e-6))
    edges = sorted(ew)
    return ig.Graph(n=N, edges=edges), [ew[e] for e in edges]


rows = []
for k in KS:
    g, w = graph(k)
    for res in RES:
        for seed in range(SEEDS):
            lab = np.array(la.find_partition(g, la.RBConfigurationVertexPartition, weights=w, resolution_parameter=res, seed=seed).membership)
            size = np.bincount(lab)
            per = []
            for grp, a, b in pairs:
                ia, ib = kix[a], kix[b]
                per.append({'group': grp, 'together': bool(lab[ia] == lab[ib]),
                            'cost_a': int(size[lab[ia]] - 1), 'cost_b': int(size[lab[ib]] - 1)})
            rows.append({'k': k, 'resolution': res, 'seed': seed, 'n_clusters': int(len(size)), 'max_cluster': int(size.max()),
                         'together': sum(p['together'] for p in per), 'mean_cost': float(np.mean([p['cost_a'] for p in per] + [p['cost_b'] for p in per])),
                         'pairs': per})
        print(k, res, 'together', [r['together'] for r in rows[-SEEDS:]], 'clusters', rows[-1]['n_clusters'], 'mean_cost', round(np.mean([r['mean_cost'] for r in rows[-SEEDS:]]), 1), flush=True)
summ = []
for k in KS:
    for res in RES:
        rs = [r for r in rows if r['k'] == k and r['resolution'] == res]
        summ.append({'k': k, 'resolution': res, 'together_mean': float(np.mean([r['together'] for r in rs])),
                     'together_per_seed': [r['together'] for r in rs], 'mean_cost': float(np.mean([r['mean_cost'] for r in rs])),
                     'n_clusters_mean': float(np.mean([r['n_clusters'] for r in rs])), 'max_cluster_mean': float(np.mean([r['max_cluster'] for r in rs])),
                     # random same-size baseline: P(partner in the query's cluster) = cost / (N-1)
                     'random_together_expected': float(3 * np.mean([r['mean_cost'] for r in rs]) / (N - 1))})
out = RUN / 'leiden'
out.mkdir(exist_ok=True)
(out / f'leiden_{TYPE}_K{K}{"" if SIM == "vn+qwen" else "_" + SIM}.json').write_text(json.dumps({'meta': {'type': TYPE, 'N': N, 'primary': {'k': 5, 'resolution': 5.0},
                                                                   'knn_sha256': hashlib.sha256(SRC.read_bytes()).hexdigest(),
                                                                   'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
                                                          'summary': summ, 'rows': rows}, ensure_ascii=False, indent=1), encoding='utf-8')
for s in summ:
    print(s['k'], s['resolution'], 'together', s['together_mean'], 'cost', round(s['mean_cost'], 1), 'random', round(s['random_together_expected'], 4), 'clusters', s['n_clusters_mean'])
