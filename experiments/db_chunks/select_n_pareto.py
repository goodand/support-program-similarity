"""Choose the number of clusters with complementary label-free criteria (사용자 선택 2026-10-28 '추천 조합').

Why: persistence (resolution plateau) and seed stability both favour coarse partitions (Ben-David et al. 2006; Good et al.
2010), so they are not complementary. Rule, fixed before looking at gold:
 grid     gamma on a log grid (G points); per gamma 10 Leiden seeds (RBConfiguration), representative = seed with the
          highest mean ARI to the others.
 filter   (pass/fail only) node-subsample stability: 5 x 80% induced subgraphs re-clustered at the same gamma, mean AMI
          to the representative on the kept nodes >= AMI_MIN; largest cluster <= MAX_SHARE of nodes; singletons <= SING_SHARE.
 A        Infomap two-level code length of the representative partition (map equation, Rosvall & Bergstrom 2008):
          shorter = the partition compresses random-walk flow better; penalises both too coarse and too fine.
 B        Significance (Traag, Krings & Van Dooren 2013; leidenalg SignificanceVertexPartition.quality) of the
          representative: how unlikely the clusters' density is under a random graph (unweighted, as defined).
 combine  Pareto set of (min A, max B) among filtered gammas; tie-break = widest run of consecutive grid gammas whose
          representative partitions are identical up to ARI >= 0.99 (persistence used only for ties).
 gold     the 3 exact-match pairs are evaluated only after the choice (never used to choose).
Parallel over gamma (process pool). Usage (this project's venv):
  .venv/Scripts/python.exe -I -X utf8 select_n_pareto.py <run> <type> <K> <k> <sim> [workers]
"""
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
G, SEEDS, SUBS, FRAC = 40, 10, 5, 0.8
GAMMA_RANGE = (0.3, 200.0)
AMI_MIN, MAX_SHARE, SING_SHARE = 0.7, 0.10, 0.20
_g = None


def load_graph(run, typ, K, k, sim):
    import igraph as ig
    src = HERE / 'runs' / run / 'knn' / (f'knn_{typ}_K{K}.npz' if sim == 'vn+qwen' else f'knn_{typ}_K{K}_{sim}.npz')
    z = np.load(src)
    N = len(z['keys'])
    pS = {(int(a), int(b)): float(s) for a, b, s in zip(z['i'], z['j'], z['S'])}
    ew = {}
    for i in range(N):  # same edge rule as leiden_eval.py / cluster_grid.run_unit
        cand = sorted(((pS[(min(i, int(j)), max(i, int(j)))], int(j)) for j in z['nbr'][i] if int(j) != i), reverse=True)[:k]
        for s, j in cand:
            e = (min(i, j), max(i, j)); ew[e] = max(ew.get(e, 0.0), max(s, 1e-6))
    edges = sorted(ew)
    g = ig.Graph(n=N, edges=edges); g.es['weight'] = [ew[e] for e in edges]
    return g, z['keys'].tolist(), src


def init(args):
    global _g
    _g = load_graph(*args)[0]


def one_gamma(gamma):
    import infomap
    import leidenalg as la
    from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
    g = _g; N = g.vcount(); w = g.es['weight']
    labs = [np.array(la.find_partition(g, la.RBConfigurationVertexPartition, weights=w, resolution_parameter=gamma, seed=s).membership)
            for s in range(SEEDS)]
    ari = np.array([[adjusted_rand_score(a, b) for b in labs] for a in labs])
    rep = labs[int(np.argmax((ari.sum(1) - 1) / (SEEDS - 1)))]
    rng = np.random.default_rng(int(gamma * 1000))
    amis = []
    for _ in range(SUBS):
        keep = np.sort(rng.choice(N, int(FRAC * N), replace=False))
        sub = g.induced_subgraph(keep.tolist())
        sl = np.array(la.find_partition(sub, la.RBConfigurationVertexPartition, weights=sub.es['weight'], resolution_parameter=gamma, seed=0).membership)
        amis.append(adjusted_mutual_info_score(rep[keep], sl))
    size = np.bincount(rep)
    im = infomap.Infomap('--two-level --silent --no-infomap')
    for e, wt in zip(g.get_edgelist(), w):
        im.add_link(e[0], e[1], wt)
    res = im.run(initial_partition={i: int(c) for i, c in enumerate(rep)})
    code = float(getattr(res, 'codelength', im.codelength))
    sig = float(la.SignificanceVertexPartition(g, initial_membership=rep.tolist()).quality())
    return {'gamma': gamma, 'N': int(len(size)), 'max_share': float(size.max() / N), 'singleton_share': float((size == 1).sum() / N),
            'seed_ARI': float((ari.sum() - SEEDS) / (SEEDS * (SEEDS - 1))), 'subsample_AMI': float(np.mean(amis)),
            'codelength': code, 'significance': sig, 'membership': rep.tolist()}


def main():
    from sklearn.metrics import adjusted_rand_score
    sys.stdout.reconfigure(encoding='utf-8')
    run, typ, K, k, sim = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    W = int(sys.argv[6]) if len(sys.argv) > 6 else 8
    g, keys, src = load_graph(run, typ, K, k, sim)
    gammas = np.exp(np.linspace(math.log(GAMMA_RANGE[0]), math.log(GAMMA_RANGE[1]), G)).tolist()
    with ProcessPoolExecutor(W, initializer=init, initargs=((run, typ, K, k, sim),)) as ex:
        pts = list(ex.map(one_gamma, gammas))
    for a, p in enumerate(pts):  # persistence = run of identical neighbours on the grid
        p['same_as_next'] = bool(a + 1 < len(pts) and adjusted_rand_score(p['membership'], pts[a + 1]['membership']) >= 0.99)
    for a, p in enumerate(pts):
        lo = a
        while lo > 0 and pts[lo - 1]['same_as_next']:
            lo -= 1
        hi = a
        while hi + 1 < len(pts) and pts[hi]['same_as_next']:
            hi += 1
        p['plateau_log_width'] = math.log(pts[hi]['gamma']) - math.log(pts[lo]['gamma'])
        p['passes'] = bool(p['subsample_AMI'] >= AMI_MIN and p['max_share'] <= MAX_SHARE and p['singleton_share'] <= SING_SHARE and p['N'] > 1)
    ok = [p for p in pts if p['passes']]
    pareto = [p for p in ok if not any((q['codelength'] <= p['codelength'] and q['significance'] >= p['significance'])
                                        and (q['codelength'] < p['codelength'] or q['significance'] > p['significance']) for q in ok)]
    chosen = max(pareto, key=lambda p: (p['plateau_log_width'], -p['codelength'])) if pareto else None
    T = json.loads((HERE / 'runs' / run / 'texts.json').read_text(encoding='utf-8'))
    kix = {x: i for i, x in enumerate(keys)}

    def gold(p):
        lab = np.array(p['membership']); size = np.bincount(lab)
        return [{'group': grp, 'together': bool(lab[kix[ks[0]]] == lab[kix[ks[1]]]),
                 'cost': [int(size[lab[kix[x]]] - 1) for x in ks]} for grp, ks in T['eval_groups'].items()]

    strip = lambda p: {x: v for x, v in p.items() if x != 'membership'}  # noqa: E731
    out = {'rule': __doc__.split('Parallel')[0], 'params': {'G': G, 'gamma_range': GAMMA_RANGE, 'seeds': SEEDS, 'subsamples': SUBS,
                                                           'frac': FRAC, 'AMI_MIN': AMI_MIN, 'MAX_SHARE': MAX_SHARE, 'SING_SHARE': SING_SHARE},
           'graph': {'type': typ, 'sim': sim, 'k': k, 'knn_sha256': hashlib.sha256(src.read_bytes()).hexdigest()},
           'chosen': strip(chosen) if chosen else None, 'pareto': [strip(p) for p in pareto],
           'post_hoc_gold_chosen': gold(chosen) if chosen else None,
           'post_hoc_gold_pareto': [{'gamma': p['gamma'], 'N': p['N'], 'gold': gold(p)} for p in pareto],
           # sensitivity of the filter threshold (reported, not used)
           'sensitivity_AMI_MIN': {str(t): sum(1 for p in pts if p['subsample_AMI'] >= t and p['max_share'] <= MAX_SHARE
                                              and p['singleton_share'] <= SING_SHARE) for t in (0.6, 0.7, 0.8)},
           'grid': [strip(p) for p in pts], 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    od = HERE / 'runs' / run / 'leiden'; od.mkdir(exist_ok=True)
    (od / f'pareto_{typ}_{sim}_k{k}.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({x: out[x] for x in ('chosen', 'post_hoc_gold_chosen', 'sensitivity_AMI_MIN')}, ensure_ascii=False, indent=1))
    print('passed', len(ok), 'of', len(pts), '| pareto', [(round(p['gamma'], 2), p['N']) for p in pareto])


if __name__ == '__main__':
    main()
