"""공고명만으로 Decision Model(pplx-decider-flash-v1.2) 순위 매기기 (668 공고 풀).

질문 문구는 실행 전에 고정했고, 평가셋을 보고 바꾸지 않는다.
각 기준 공고 A마다 나머지 667개 B에 noul 질문 1개씩 보내고, 점수 = logit(true) - logit(false) (모델 카드가 순위용으로 권하는 값).
행 단위 캐시(npz)로 중단 후 재개. 참조 경로(fast=False), 난수 없음.

Usage:
  <DECIDER_ENV_PYTHON> -I -X utf8 runs/044/decision_model_harness/decider_rank_harness.py run
  <DECIDER_ENV_PYTHON> -I -X utf8 runs/044/decision_model_harness/decider_rank_harness.py eval
"""
import hashlib, json, os, sys, time
from pathlib import Path
import numpy as np

R = Path(os.environ.get('DEC_ROOT', '<DEV_EVAL_ROOT>/jh_dev/<EXPERIMENT_DIR>'))
INP = R / 'runs/044/pool668_newreview_harness/texts.json'
OUT = R / 'runs/044/decision_model_harness'
CK = Path(os.environ.get('DEC_CK', '<DEV_EVAL_ROOT>/_models_harness/pplx-decider-flash-v1.2'))
REV = os.environ.get('DEC_REV', '9ff7cf9838e59df461c276b7699d56b244d2af58')
TAG = os.environ.get('DEC_TAG', '')  # 빈 값 = flash(기존 파일명), 예: '_27b'
QUESTION = {'type': 'noul',
            'instructions': '공고 A와 공고 B는 지원 목적과 지원 대상이 비슷한 유사 사업인가?',
            'criteria': {'true': '유사한 사업이다', 'false': '유사한 사업이 아니다'}}
CACHE = OUT / f'rows_cache{TAG}_harness'
BATCH = int(os.environ.get('DEC_BATCH', '667'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load():
    T = json.loads(INP.read_text(encoding='utf-8'))
    keys = T['keys']; tx = T['texts']['chunk']
    if os.environ.get('DEC_INPUT', 'title') == 'full9':  # 9칸 전체 청크 텍스트 그대로(기준선 임베딩 입력과 같은 문자열)
        titles = [tx[k] for k in keys]
    else:
        titles = [tx[k].split(' / 기관: ')[0].removeprefix('26공고명: ').strip() for k in keys]
    assert all(tx[k].startswith('26공고명: ') for k in keys) and len(keys) == 668
    return T, keys, titles


def run():
    sys.path.insert(0, str(CK / 'source' / 'src'))
    import torch
    from pplx_decider import Decider
    _, keys, titles = load()
    CACHE.mkdir(parents=True, exist_ok=True)
    dec = Decider.from_pretrained(CK, device='cuda')
    N = len(keys); t0 = time.time(); done = 0
    T = json.loads(INP.read_text(encoding='utf-8')); kix = {k: i for i, k in enumerate(keys)}
    ev = sorted({kix[x] for pr in T['eval_groups'].values() for x in pr})  # 평가 질의 35행 먼저(결과는 순서와 무관)
    for a in ev + [i for i in range(N) if i not in set(ev)]:
        f = CACHE / f'row_{a:03d}.npy'
        if f.exists():
            continue
        reqs = [{'state': json.dumps({'공고 A': titles[a], '공고 B': titles[b]}, ensure_ascii=False),
                 'questions': {'similar': QUESTION}} for b in range(N) if b != a]
        out = []
        for s in range(0, len(reqs), BATCH):
            for lg in dec.logits(reqs[s:s + BATCH]):
                v = lg['similar']; assert len(v) == 2; out.append(v[1] - v[0])  # protocol.py:68 옵션 순서: false, true
        row = np.full(N, np.nan, dtype=np.float64); row[[b for b in range(N) if b != a]] = out
        tmp = f.with_suffix('.tmp.npy'); np.save(tmp, row); os.replace(tmp, f)
        done += 1
        if done == 1 or done % 20 == 0:
            el = time.time() - t0
            print(f'row {a} done | {done} rows in {el:.0f}s | gpu {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB', flush=True)
    M = np.stack([np.load(CACHE / f'row_{a:03d}.npy') for a in range(N)])
    np.savez_compressed(OUT / f'decider_scores_pool668{TAG}_harness.npz', score=M, keys=np.array(keys), titles=np.array(titles))
    print('saved matrix', M.shape, flush=True)


def evaluate():
    T, keys, titles = load()
    kix = {k: i for i, k in enumerate(keys)}
    mp = OUT / f'decider_scores_pool668{TAG}_harness.npz'
    if mp.exists():
        M = np.load(mp)['score']
    else:  # 평가 질의 행만 계산한 경우: 캐시 행만 읽고 나머지 행은 NaN (지표는 질의 행만 씀)
        M = np.full((len(keys), len(keys)), np.nan)
        for a in range(len(keys)):
            f = CACHE / f'row_{a:03d}.npy'
            if f.exists():
                M[a] = np.load(f)
        ev = {kix[x] for pr in T['eval_groups'].values() for x in pr}
        assert all((CACHE / f'row_{a:03d}.npy').exists() for a in ev), '평가 질의 행이 빠짐'
    N = len(keys)

    def rank(a, b):  # eval_v3.py rank()와 같은 규칙: 1 + 더 큰 수 + (동점 수 - 1)/2, 자기 자신 제외 (공고만 풀이라 별칭 없음)
        r = M[a].copy(); r[a] = -np.inf; r = np.nan_to_num(r, nan=-np.inf)
        v = r[b]; return 1 + int((r > v).sum()) + ((r == v).sum() - 1) / 2

    pairs = []
    for name, (x, y) in T['eval_groups'].items():
        i, j = kix[x], kix[y]
        pairs.append({'pair': name, 'group': name.split('#')[0], 'a': x, 'b': y,
                      'rank_b_from_a': rank(i, j), 'rank_a_from_b': rank(j, i),
                      'score_ab': float(M[i, j]), 'score_ba': float(M[j, i])})
    R = [p['rank_b_from_a'] for p in pairs] + [p['rank_a_from_b'] for p in pairs]
    groups = {}
    for name, (x, y) in T['eval_groups'].items():
        groups.setdefault(name.split('#')[0], set()).update([x, y])
    last = {}  # 기준 공고마다 같은 그룹 짝 전부를 찾는 마지막 순위
    for g, mem in groups.items():
        for x in mem:
            last[x] = max(rank(kix[x], kix[y]) for y in mem if y != x)
    done_rows = [a for a in range(N) if not np.isnan(M[a]).all()]
    ties = int(sum(len(np.unique(M[a][~np.isnan(M[a])])) < N - 1 for a in done_rows))
    summ = {'n_pairs': len(pairs), 'n_directed': len(R), 'n_groups': len(groups), 'n_query_docs': len(last),
            'top5_share': sum(r <= 5 for r in R) / len(R), 'top30_share': sum(r <= 30 for r in R) / len(R),
            'rank_median': float(np.median(R)), 'rank_max': float(max(R)),
            'last_correct_rank_max': float(max(last.values())), 'last_correct_rank_median': float(np.median(list(last.values()))),
            'rows_with_tied_scores': ties, 'rows_computed': len(done_rows)}
    L = np.array(sorted(last.values()), float)  # score_pool668_newreview_harness.py summ()과 같은 항목
    summ['baseline_style'] = {'n': len(L), 'logmean': float(np.log(L).mean()), 'mean': float(L.mean()), 'median': float(np.median(L)),
                              'in100': float((L <= 100).mean()), 'worst': float(L.max())}
    meta = {'model': json.loads((CK / 'release-manifest.json').read_text(encoding='utf-8')).get('repo_id'), 'revision': REV,
            'weights_sha256': {p.name: sha(p) for p in sorted(CK.glob('*.safetensors'))},
            'input': str(INP), 'input_sha256': sha(INP), 'question': QUESTION,
            'input_mode': os.environ.get('DEC_INPUT', 'title'),
            'state_format': '{"공고 A": 기준 공고명, "공고 B": 후보 공고명} JSON 문자열',
            'title_rule': "texts['chunk'][key]를 ' / 기관: ' 앞까지 자르고 '26공고명: ' 접두와 앞뒤 공백 제거",
            'score': 'logit(true) - logit(false), 온도 적용 전', 'path': 'reference (fast=False), 난수 없음',
            'rank_rule': 'eval_v3.py rank()와 같음: 1 + 더 큰 후보 수 + (동점 수 - 1)/2, 자기 자신 제외',
            'commands': ['<DECIDER_ENV_PYTHON> -I -X utf8 runs/044/decision_model_harness/decider_rank_harness.py run',
                         '<DECIDER_ENV_PYTHON> -I -X utf8 runs/044/decision_model_harness/decider_rank_harness.py eval'],
            'code_sha256': sha(__file__)}
    (OUT / f'decider_eval_pool668{TAG}_harness.json').write_text(json.dumps({'summary': summ, 'meta': meta, 'pairs': pairs, 'last_correct_rank': last},
                                                                       ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summ, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    {'run': run, 'eval': evaluate}[sys.argv[1]]()
