"""메인 세션 eval_v3.py(run 011)를 그대로 실행하고 Qwen 임베딩만 교체해 채점. mm의 Qwen min/max는 새 임베딩으로 재계산. 결과는 출력만(메인 run 폴더에 쓰지 않음).
usage (호스트): <EMBED_ENV_PYTHON> -I -X utf8 code/score_ft.py <npz> [q:<npz> ...]  — base도 함께 출력(재현 확인용)
'q:' 접두 = 질의 쪽만 그 임베딩, 후보(문서) 쪽은 base(비대칭 instruction 비교용). 없으면 양쪽 같은 임베딩(대칭)."""
import sys, numpy as np, torch
from pathlib import Path
EXP = Path('<DEV_EVAL_ROOT>/jh_dev/<EXPERIMENT_DIR>')
src = (EXP / 'eval_v3.py').read_text(encoding='utf-8')
sys.argv_ft = sys.argv[1:]; sys.argv = ['eval_v3.py', '011', '30']
g = {'__file__': str(EXP / 'eval_v3.py'), '__name__': 'core'}
exec(src[:src.index('cache = {}')], g)
R, N, ev, ix, rank = g['R'], g['N'], g['ev'], g['ix'], g['rank']
_f = src[src.index('def rows_of'):src.index('# twin alias')]
assert 'qc = E @ E[q]' in _f; exec(_f.replace('qc = E @ E[q]', 'qc = E @ EQ[q]'), g)  # rows_of 원문 그대로, 질의 벡터만 EQ로
def rowmm(E, Q=None):  # 행 q: Q[q]·E 의 min/max(자기 자신 제외)
    X = torch.tensor(E, device='cuda'); Y = X if Q is None else torch.tensor(Q, device='cuda'); lo, hi = torch.empty(N, device='cuda'), torch.empty(N, device='cuda')
    for s in range(0, N, 4096):
        B = Y[s:s+4096] @ X.T; d = torch.arange(s, min(s+4096, N), device='cuda'); r = torch.arange(len(d), device='cuda')
        B[r, d] = float('inf'); lo[s:s+len(d)] = B.min(1).values; B[r, d] = float('-inf'); hi[s:s+len(d)] = B.max(1).values
    return lo.cpu().numpy(), hi.cpu().numpy()
def score(tag, E, Q=None):
    g['E'], g['EQ'] = E, (E if Q is None else Q); mm = g['mm'].copy(); mm[:, 2], mm[:, 3] = rowmm(E, Q); g['mm'] = mm
    out = {}
    for _, a, b in ev['pairs']:
        for q, p in ((ix[a], ix[b]), (ix[b], ix[a])):
            rows = g['rows_of'](q)
            for n, s in rows.items(): out.setdefault(n, []).append(rank(s, q, p))
    for n in ('Qwen', 'VN+Qwen'):
        rk = np.array(out[n]); print(f"{tag:10s} {n:8s} MRR {np.mean(1/rk):.3f} R@1 {np.mean(rk<=1):.2f} R@10 {np.mean(rk<=10):.2f} R@30 {np.mean(rk<=30):.2f} R@100 {np.mean(rk<=100):.2f} med {np.median(rk):.1f} p90 {np.percentile(rk,90):.0f} max {rk.max()}", flush=True)
    return {n: np.array(v) for n, v in out.items()}
def boot(a, b, B=10000, seed=0):  # 쌍 단위 paired bootstrap: 51쌍을 복원추출(양방향 질의는 함께), MRR 차이 b-a의 95% 구간
    ra, rb = 1 / a.reshape(-1, 2), 1 / b.reshape(-1, 2); d = (rb - ra).mean(1); rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    return d.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), (bs > 0).mean()
base = g['E'].copy(); mm0 = g['mm'].copy()
rb = score('base', base); g['mm'] = mm0; runs = {}
for arg in sys.argv_ft:
    qonly = arg.startswith('q:'); path = arg[2:] if qonly else arg
    with np.load(path) as z: emb = dict(zip(z['h'].tolist(), z['v']))
    X = np.stack([emb[r['sha256']] for r in R]).astype(np.float32); tag = ('q:' if qonly else '') + Path(path).parent.name[:24]
    g['mm'] = mm0.copy(); r1 = score(tag, base, X) if qonly else score(tag, X); runs[tag] = (path, r1)
    for n in ('Qwen', 'VN+Qwen'):
        m, lo, hi, pp = boot(rb[n], r1[n]); print(f"   vs base {n:8s} dMRR {m:+.3f}  95% [{lo:+.3f}, {hi:+.3f}]  P(개선) {pp:.2f}", flush=True)
# 여러 후보를 한 번에 볼 때 '우연 통과' 보정: 같은 복원추출 표본을 모든 후보에 공유(joint bootstrap)해,
# 효과 0일 때 후보들 중 최댓값이 얼마나 커지는지(max-T)로 동시 P를 낸다. 후보 1개면 보정 없음과 같다.
import json, time, hashlib
LEDGER = Path(__file__).resolve().parents[1] / 'logs' / 'screen_ledger.jsonl'
# 지표: MRR(1/순위), logR(-log 순위: 뒤쪽 순위 개선도 같은 비중 → 마지막 정답 순위·검토 비용 쪽). 질의당 정답이 하나라 MAP = MRR.
MET = {'MRR': lambda r: 1 / r, 'logR': lambda r: -np.log(r)}
for n, met in [(n, k) for k in MET for n in ('Qwen', 'VN+Qwen')]:
    if not runs: break
    f = MET[met]; tags = list(runs); D = np.stack([(f(runs[t][1][n].reshape(-1, 2)) - f(rb[n].reshape(-1, 2))).mean(1) for t in tags])  # (후보, 51쌍)
    idx = np.random.default_rng(0).integers(0, D.shape[1], (10000, D.shape[1]))
    BS = D[:, idx].mean(2)                                     # (후보, B) 각 후보의 bootstrap 평균 차이
    obs = D.mean(1); sd = BS.std(1) + 1e-12; T0 = (BS - obs[:, None]) / sd[:, None]  # 효과 0으로 옮긴 귀무 분포
    maxT = T0.max(0)
    print(f"[동시 보정 {n} {met}] 후보 {len(tags)}개 (기록: {LEDGER.name})")
    for i, t in enumerate(tags):
        p_one = float((T0[i] >= obs[i] / sd[i]).mean()); p_fw = float((maxT >= obs[i] / sd[i]).mean())
        print(f"   {t:30s} d{met} {obs[i]:+.3f}  P(개선) {(BS[i] > 0).mean():.2f}  p(단독) {p_one:.3f}  p(동시) {p_fw:.3f}", flush=True)
        LEDGER.parent.mkdir(exist_ok=True)
        with open(LEDGER, 'a', encoding='utf-8') as f:
            f.write(json.dumps({'at': time.strftime('%Y-%m-%d %H:%M:%S'), 'tag': t, 'npz': runs[t][0], 'sim': n, 'metric': met, 'batch_size': len(tags),
                                'd': round(float(obs[i]), 4), 'dMRR': round(float(obs[i]), 4) if met == 'MRR' else None, 'P_improve': round(float((BS[i] > 0).mean()), 3),
                                'p_single': round(p_one, 4), 'p_familywise': round(p_fw, 4),
                                'eval_v3_sha256': hashlib.sha256((EXP / 'eval_v3.py').read_bytes()).hexdigest()}, ensure_ascii=False) + '\n')
