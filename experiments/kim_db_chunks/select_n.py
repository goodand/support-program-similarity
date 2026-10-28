"""Choose the number of clusters N with an explainable rule (사용자 요청 2026-10-28: 성능과 설명가능성).

Rule (fixed before looking at gold):
 1. Resolution profile (Traag, Krings & Van Dooren 2013; leidenalg Optimiser.resolution_profile): bisection over the
    resolution finds every point where the optimal partition changes -> a staircase of partitions, each valid on an
    interval [r_lo, r_hi).
 2. Constraint: largest cluster <= M (review budget; config, default 50).
 3. Among admissible partitions, pick the one valid over the widest interval in log(resolution) (most persistent).
 4. Report its N, interval, seed stability (ARI of 10 seeds at the interval's geometric midpoint) and, only afterwards,
    gold pairs together / review cost (post-hoc check; gold is never used to choose).
Graph = the same kNN graph as leiden_eval.py (edge rule of cluster_grid.run_unit).
Usage: <DEV_EVAL_ROOT>/jh_dev/program_mapping_nlp_v1/.venv/Scripts/python.exe -X utf8 select_n.py <run> <type> <K> <k> [sim] [M]
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import igraph as ig
import leidenalg as la
import numpy as np
from sklearn.metrics import adjusted_rand_score

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs' / sys.argv[1]
TYPE, K, k = sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
SIM = sys.argv[5] if len(sys.argv) > 5 else 'vn+qwen'
M = int(sys.argv[6]) if len(sys.argv) > 6 else 50
# 2026-10-28: the full range 0.05-200 spent >70 min on low resolutions whose partitions can never satisfy max cluster <= M.
# Search only [R_LO, R_HI]; the rule is unchanged. If no admissible partition is found, the run says so (widen R_LO).
R_LO = float(sys.argv[7]) if len(sys.argv) > 7 else 10.0
R_HI = float(sys.argv[8]) if len(sys.argv) > 8 else 200.0
SRC = RUN / 'knn' / (f'knn_{TYPE}_K{K}.npz' if SIM == 'vn+qwen' else f'knn_{TYPE}_K{K}_{SIM}.npz')
z = np.load(SRC)
keys = z['keys'].tolist(); N = len(keys); kix = {x: i for i, x in enumerate(keys)}
pS = {(int(a), int(b)): float(s) for a, b, s in zip(z['i'], z['j'], z['S'])}
nbr = z['nbr']
ew = {}
for i in range(N):
    cand = sorted(((pS[(min(i, int(j)), max(i, int(j)))], int(j)) for j in nbr[i] if int(j) != i), reverse=True)[:k]
    for s, j in cand:
        e = (min(i, j), max(i, j)); ew[e] = max(ew.get(e, 0.0), max(s, 1e-6))
edges = sorted(ew)
g = ig.Graph(n=N, edges=edges); w = [ew[e] for e in edges]
g.es['weight'] = w

prof = la.Optimiser().resolution_profile(g, la.RBConfigurationVertexPartition, resolution_range=(R_LO, R_HI), weights='weight',
                                         number_iterations=-1, linear_bisection=False)
steps = []
for a, p in enumerate(prof):
    lo = p.resolution_parameter
    hi = prof[a + 1].resolution_parameter if a + 1 < len(prof) else R_HI
    size = np.bincount(np.array(p.membership))
    steps.append({'r_lo': lo, 'r_hi': hi, 'width_log': math.log(hi) - math.log(lo), 'N': int(len(size)), 'max_cluster': int(size.max()),
                  'admissible': bool(size.max() <= M), 'membership': p.membership})
# M = 0 -> no size constraint (사용자 2026-10-28: 안정성·지속성만으로 결정). Trivial partitions (N = 1) are never admissible.
for st in steps:
    st['admissible'] = bool(st['N'] > 1 and (M == 0 or st['max_cluster'] <= M))
adm = [s for s in steps if s['admissible']]
assert adm, f'no partition with max cluster <= {M} in resolution [{R_LO}, {R_HI}] -> widen the range'


def seed_stability(st):
    mid = math.sqrt(st['r_lo'] * st['r_hi'])
    labs = [np.array(la.find_partition(g, la.RBConfigurationVertexPartition, weights=w, resolution_parameter=mid, seed=s).membership) for s in range(10)]
    return [adjusted_rand_score(labs[a], labs[b]) for a in range(10) for b in range(a + 1, 10)]


if M == 0:
    # persistence shortlists the 5 widest intervals; stability (mean seed ARI) decides among them
    short = sorted(adm, key=lambda s: -s['width_log'])[:5]
    for st in short:
        st['seed_ARI'] = seed_stability(st)
    best = max(short, key=lambda s: float(np.mean(s['seed_ARI'])))
    aris = best['seed_ARI']
    shortlist = [{'N': s['N'], 'interval': [s['r_lo'], s['r_hi']], 'width_log': s['width_log'], 'max_cluster': s['max_cluster'],
                  'seed_ARI_mean': float(np.mean(s['seed_ARI']))} for s in short]
else:
    best = max(adm, key=lambda s: s['width_log'])
    aris = seed_stability(best)
    shortlist = []
T = json.loads((RUN / 'texts.json').read_text(encoding='utf-8'))
lab = np.array(best['membership']); size = np.bincount(lab)
post = [{'group': grp, 'together': bool(lab[kix[ks[0]]] == lab[kix[ks[1]]]), 'cost': [int(size[lab[kix[x]]] - 1) for x in ks]}
        for grp, ks in T['eval_groups'].items()]
res = {'rule': 'widest log-resolution interval among partitions with max cluster <= M', 'M': M, 'search_range': [R_LO, R_HI], 'type': TYPE, 'sim': SIM, 'k': k,
       'chosen': {'N': best['N'], 'resolution_interval': [best['r_lo'], best['r_hi']], 'width_log': best['width_log'],
                  'max_cluster': best['max_cluster'], 'seed_ARI_mean': float(np.mean(aris)), 'seed_ARI_min': float(np.min(aris))},
       'shortlist_persistence_then_stability': shortlist, 'post_hoc_gold': post, 'random_together_expected': float(sum(np.mean(p['cost']) for p in post) / (N - 1)),
       'profile': [{x: s[x] for x in s if x not in ('membership', 'seed_ARI')} for s in steps],
       'knn_sha256': hashlib.sha256(SRC.read_bytes()).hexdigest(), 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
out = RUN / 'leiden'; out.mkdir(exist_ok=True)
(out / f'select_n_{TYPE}_{SIM}_k{k}_M{M}.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({x: res[x] for x in ('chosen', 'post_hoc_gold', 'random_together_expected')}, ensure_ascii=False, indent=1))
print('profile steps', len(steps), 'admissible', len(adm))
