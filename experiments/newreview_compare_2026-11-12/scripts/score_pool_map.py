"""runs/036 (2026-11-06 야간, GPU 없음): 한 풀(run 011 또는 035)에서 run별 질의 단위 점수를 계산해 npz로 저장.
eval_v3.py 앞부분을 exec해 풀·VN·별칭·rank를 그대로 쓰고(decide_candidates.py와 같은 방식), 임베딩만 run 것으로 바꾼다(Qwen 행 min/max는 run 임베딩으로 재계산).
질의 = 사람판정 평가셋(test, eval_set_v2 그룹 원소 각각, 그룹 안 다른 원소가 정답) + 규칙 선별셋(dev v0 433).
질의마다: last = 마지막 정답 순위, ap = 평균정밀도, VN+Qwen과 Qwen 두 점수 방식.
usage: <EMBED_ENV_PYTHON> -I -X utf8 runs/036/score_pool.py <pool 011|035> <out_dir> <name=npz[+npz…]> [...]
  npz를 +로 이으면 h 키로 합침(035 풀: qwen_chunk.npz + qwen_new2_034.npz). 출력 <out_dir>/<pool>__<name>.npz(이미 있으면 건너뜀, 원자적 저장)."""
import json, os, sys, time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
sys.stdout.reconfigure(encoding='utf-8')
KIM = Path(__file__).resolve().parents[2]
POOL, OUTD, ITEMS = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
assert POOL in ('011', '035'); OUTD.mkdir(parents=True, exist_ok=True)
src = (KIM / 'eval_v3.py').read_text(encoding='utf-8'); sys.argv = ['eval_v3.py', POOL, '30']
g = {'__file__': str(KIM / 'eval_v3.py'), '__name__': 'core'}; exec(src[:src.index('cache = {}')], g)
R, ix, ev, rank, N = g['R'], g['ix'], g['ev'], g['rank'], g['N']
DEV = json.loads((KIM / 'runs' / '017' / 'dev_v0.json').read_text(encoding='utf-8'))['groups']
TEST = defaultdict(set)
for gr, a, b in ev['pairs']:
    TEST[gr] |= {a, b}
SETS = {'test': {gr: [ix[k] for k in sorted(ks)] for gr, ks in sorted(TEST.items())}, 'dev': {gid: [ix[k] for k in ks] for gid, ks in sorted(DEV.items())}}
QS = {s: sorted((gid, q) for gid, mem in SETS[s].items() for q in mem) for s in SETS}
mm0 = g['mm'].copy()
def rowmm(E):
    X = torch.tensor(E); lo, hi = torch.empty(N), torch.empty(N)
    for a in range(0, N, 2048):
        B = X[a:a + 2048] @ X.T; d = torch.arange(a, min(a + 2048, N)); r_ = torch.arange(len(d))
        B[r_, d] = float('inf'); lo[a:a + len(d)] = B.min(1).values; B[r_, d] = float('-inf'); hi[a:a + len(d)] = B.max(1).values
    return lo.numpy(), hi.numpy()
def load(spec):
    emb = {}
    for p in spec.split('+'):
        with np.load(p) as z:
            emb.update(zip(z['h'].tolist(), z['v']))
    miss = sum(r['sha256'] not in emb for r in R)
    if miss:
        raise ValueError(f'{spec}: 풀 {POOL} 청크 {N}개 중 {miss}개 임베딩 없음')
    return np.stack([emb[r['sha256']] for r in R]).astype(np.float32)
for it in ITEMS:
    name, spec = it.split('=', 1); out = OUTD / f'{POOL}__{name}.npz'
    if out.exists():
        print('skip', out.name, flush=True); continue
    t0 = time.time()
    try:
        X = load(spec)
    except ValueError as e:  # 불완전 npz(시험·부분 인코딩)는 이 항목만 건너뜀(11-06 일괄 채점이 64건짜리 탐침 npz에서 멈춘 것 재발 방지)
        print('skip(불완전)', name, e, flush=True); continue
    g['E'] = X; mm = mm0.copy(); mm[:, 2], mm[:, 3] = rowmm(X); g['mm'] = mm
    res = {}
    for s in SETS:
        for sim in ('VN+Qwen', 'Qwen'):
            last, ap = [], []
            for gid, q in QS[s]:
                sc = np.array(g['rows_of'](q)[sim], float); sc[q] = np.nan
                rk = sorted(rank(sc, q, o) for o in SETS[s][gid] if o != q); assert rk, f'{s} 그룹 {gid}에 정답 없음'
                last.append(rk[-1]); ap.append(float(np.mean([(k + 1) / r for k, r in enumerate(rk)])))
            res[f'{s}__{sim}__last'] = np.array(last); res[f'{s}__{sim}__ap'] = np.array(ap)
        res[f'{s}__groups'] = np.array([gid for gid, _ in QS[s]]); res[f'{s}__qkeys'] = np.array([g['keys'][q] for _, q in QS[s]])  # 풀 간 정렬용(적대 검증 #3)
    tmp = out.with_suffix('.tmp.npz'); np.savez(tmp, pool=POOL, spec=spec, n_pool=N, **res); os.replace(tmp, out)
    print(f"{out.name} test VN+Qwen logmean {np.log(res['test__VN+Qwen__last']).mean():.3f} worst(Qwen) {res['test__Qwen__last'].max():.0f} {time.time() - t0:.0f}s", flush=True)
