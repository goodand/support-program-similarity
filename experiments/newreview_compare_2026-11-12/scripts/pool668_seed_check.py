"""668 공고 그래프, 해상도 13.88(격자 값 그대로)에서 seed 0~99 가장 큰 군집·그룹 포함."""
import sys, json, numpy as np
from collections import Counter, defaultdict
sys.path.insert(0, '.')
import select_n_pareto as S
import leidenalg as la
B = 'runs/044/pool668_newreview_harness/'
M = np.load(B + 'leiden/membership_chunk_vn+qwen_k5.npz')
gam = float([g for g in M['gamma'] if abs(g - 13.88) < 0.01][0])
g, keys, _ = S.load_graph('044/pool668_newreview_harness', 'chunk', 30, 5, 'vn+qwen')
kix = {k: i for i, k in enumerate(keys)}
T = json.load(open(B + 'texts.json', encoding='utf-8'))
grp = defaultdict(set)
for n, (x, y) in T['eval_groups'].items():
    grp[n.split('#')[0]].update([x, y])
rows = []
for s in range(100):
    lab = np.array(la.find_partition(g, la.RBConfigurationVertexPartition, weights=g.es['weight'], resolution_parameter=gam, seed=s).membership)
    size = np.bincount(lab)
    rows.append((int(size.max()), len(size), sum(len({lab[kix[x]] for x in v}) == 1 for v in grp.values())))
print('gamma', gam)
print('largest', sorted(Counter(r[0] for r in rows).items()))
print('N', min(r[1] for r in rows), max(r[1] for r in rows))
print('groups_complete', sorted(Counter(r[2] for r in rows).items()))
print("cross", sorted(Counter((r[0], r[2]) for r in rows).items()))
