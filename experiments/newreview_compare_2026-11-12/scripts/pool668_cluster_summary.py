"""668 공고만 군집화한 기준선 요약 (select_n_pareto.py 결과 읽기만)."""
import json, numpy as np
from collections import defaultdict
B = 'runs/044/pool668_newreview_harness/'
P = json.load(open(B + 'leiden/pareto_chunk_vn+qwen_k5.json', encoding='utf-8'))
M = np.load(B + 'leiden/membership_chunk_vn+qwen_k5.npz')
T = json.load(open(B + 'texts.json', encoding='utf-8'))
keys = list(M['keys']); kix = {k: i for i, k in enumerate(keys)}
groups = defaultdict(set)
for name, (x, y) in T['eval_groups'].items():
    groups[name.split('#')[0]].update([x, y])
out = {'chosen_gamma': float(M['chosen_gamma']), 'pareto': [(round(p['gamma'], 2), p['N']) for p in P['pareto']], 'rows': []}
for gi, g in enumerate(M['gamma']):
    lab = M['membership'][gi]; size = np.bincount(lab)
    comp = sum(len({lab[kix[x]] for x in s}) == 1 for s in groups.values())
    pairs = [(lab[kix[x]] == lab[kix[y]], size[lab[kix[x]]] - 1, size[lab[kix[y]]] - 1) for x, y in T['eval_groups'].values()]
    out['rows'].append({'gamma': round(float(g), 2), 'N': int(M['N'][gi]), 'largest': int(size.max()),
                        'top5': sorted(size.tolist())[-5:][::-1], 'median_size': float(np.median(size)),
                        'groups_complete': f'{comp}/{len(groups)}', 'pairs_together': f'{sum(p[0] for p in pairs)}/{len(pairs)}',
                        'cost_mean': round(float(np.mean([c for p in pairs for c in p[1:]])), 2),
                        'chosen': bool(np.isclose(g, M['chosen_gamma']))})
json.dump(out, open(B + 'cluster_summary_harness.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
for r in out['rows']:
    print(r)
