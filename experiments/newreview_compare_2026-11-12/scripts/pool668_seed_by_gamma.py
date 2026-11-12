"""668 공고 그래프: 격자 해상도 40개 x seed 0~99, 그룹 완전 포함 14/14 비율과 가장 큰 군집."""
import sys, json, numpy as np
from collections import defaultdict
sys.path.insert(0, '.')
import select_n_pareto as S
import leidenalg as la
B = 'runs/044/pool668_newreview_harness/'
M = np.load(B + 'leiden/membership_chunk_vn+qwen_k5.npz')
g, keys, _ = S.load_graph('044/pool668_newreview_harness', 'chunk', 30, 5, 'vn+qwen')
kix = {k: i for i, k in enumerate(keys)}
T = json.load(open(B + 'texts.json', encoding='utf-8'))
grp = defaultdict(set)
for n, (x, y) in T['eval_groups'].items():
    grp[n.split('#')[0]].update([x, y])
out = []
for gam in M['gamma']:
    full, big = 0, []
    for s in range(100):
        lab = np.array(la.find_partition(g, la.RBConfigurationVertexPartition, weights=g.es['weight'], resolution_parameter=float(gam), seed=s).membership)
        full += all(len({lab[kix[x]] for x in v}) == 1 for v in grp.values()); big.append(int(np.bincount(lab).max()))
    r = {'gamma': round(float(gam), 2), 'full14': full, 'largest_min': min(big), 'largest_med': float(np.median(big)), 'largest_max': max(big)}
    out.append(r); print(r, flush=True)
json.dump(out, open(B + 'seed_by_gamma_harness.json', 'w', encoding='utf-8'), indent=1)
