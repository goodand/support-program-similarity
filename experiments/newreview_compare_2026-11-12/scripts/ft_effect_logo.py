"""그룹 하나씩 빼기(무작위 없음): 학습 전 vs 5스텝 학습(seed 42·43·45·46 평균)의 평균 로그 순위 변화와 가장 나쁜 순위가
그룹 14개를 하나씩 뺄 때 어떻게 바뀌는지. 가장 나쁜 순위를 만드는 그룹도 표시.
입력: ft_effect_paired_harness.py와 같은 질의별 마지막 정답 순위(공고명 단독, 청크 단독, 청크 결합).
usage: <EMBED_ENV_PYTHON> -I -X utf8 runs/044/ft_effect_logo_harness.py
출력: runs/044/ft_effect_logo_harness.json"""
import json, sys, hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.stdout.reconfigure(encoding='utf-8')
KIM = Path(__file__).resolve().parents[2]
EVF = KIM / 'runs/044/pool668_newreview_harness/eval_set_pool668.json'
src = (KIM / 'eval_v3.py').read_text(encoding='utf-8'); src = src.replace("EVAL = HERE / 'runs' / '009' / 'eval_set_v2.json'", f"EVAL = Path({str(EVF)!r})")
sys.argv = ['eval_v3.py', '044/pool668_newreview_harness', '30']; g = {'__file__': str(KIM / 'eval_v3.py'), '__name__': 'core'}; exec(src[:src.index('cache = {}')], g)
R, ix, ev, rank = g['R'], g['ix'], g['ev'], g['rank']
C = [json.loads(l) for l in open(KIM / 'runs/044/pool668_newreview_harness/chunks.jsonl', encoding='utf-8')]
T = defaultdict(set)
for gr, a, b in ev['pairs']: T[gr] |= {ix[a], ix[b]}
QS = sorted((gr, q) for gr, m in T.items() for q in m); GR = np.array([gr for gr, _ in QS])
def last_from(X):
    S = X @ X.T; L = []
    for gr, q in QS:
        sc = S[q].astype(float).copy(); sc[q] = np.nan; L.append(max(rank(sc, q, o) for o in T[gr] if o != q))
    return np.array(L, float)
def name_X(f):
    with np.load(f) as z: e = dict(zip(z['h'].tolist(), z['v']))
    return np.stack([e[hashlib.sha256(c['name'].encode()).hexdigest()] for c in C]).astype(np.float32)
NE = KIM / 'runs/044/name_enc_harness'
nb = last_from(name_X(NE / 'qwen_name_base.npz'))
na = {42: last_from(name_X(NE / 'qwen_name_b3b128.npz'))}
for s in (43, 45, 46): na[s] = last_from(name_X(NE / f'qwen_name_b3b128_s{s}.npz'))
b9 = json.load(open(KIM / 'runs/044/newreview_scores_pool668_q668_harness.json', encoding='utf-8'))['runs']['학습 전']
rep = json.load(open(KIM / 'runs/044/seed_rep_harness/seed_rep_scores_harness.json', encoding='utf-8'))['runs']
sets = {'공고명 단독': (nb, na),
        '청크 단독': (np.array(b9['Qwen']['last'], float), {s: np.array(rep[f'seed {s}']['Qwen']['last'], float) for s in (42, 43, 45, 46)}),
        '청크 결합': (np.array(b9['VN+Qwen']['last'], float), {s: np.array(rep[f'seed {s}']['VN+Qwen']['last'], float) for s in (42, 43, 45, 46)})}
groups = sorted(set(GR)); out = {}
for k, (bef, aft) in sets.items():
    A = np.stack(list(aft.values()))  # seed x 질의
    d = np.mean(np.log(A) - np.log(bef), 0)
    rows = {}
    for gg in ['(없음)'] + groups:
        m = GR != gg
        rows[gg] = {'mean_dlog': float(d[m].mean()), 'worst_before': float(bef[m].max()),
                    'worst_after_seeds': [float(x) for x in A[:, m].max(1)]}
    wb = GR[np.argmax(bef)]; wa = sorted({GR[np.argmax(a)] for a in A})
    per_group = {gg: {'n_q': int((GR == gg).sum()), 'dlog_sum': float(d[GR == gg].sum()),
                      'last_before': bef[GR == gg].tolist(), 'last_after_mean': A[:, GR == gg].mean(0).round(2).tolist()} for gg in groups}
    out[k] = {'leave_one_out': rows, 'worst_group_before': wb, 'worst_group_after': wa, 'per_group': per_group}
    print(f'\n== {k}: 가장 나쁜 순위 그룹 학습 전 {wb}, 학습 후(seed별) {wa}')
    print(f"{'뺀 그룹':8s} {'평균Δlog':>9s} {'최악 전':>7s} {'최악 후(seed)':>18s}  그룹 Δ합  질의 수")
    for gg, r in rows.items():
        pg = per_group.get(gg, {})
        print(f"{gg:8s} {r['mean_dlog']:+9.3f} {r['worst_before']:7.0f} {str([int(x) for x in r['worst_after_seeds']]):>18s}  {pg.get('dlog_sum', 0):+6.2f}  {pg.get('n_q', '')}")
json.dump(out, open(KIM / 'runs/044/ft_effect_logo_harness.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
