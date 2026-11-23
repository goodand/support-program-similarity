"""방법별 성능 비교 엑셀 (668 풀, 평가셋 질의 35개). 입력 조건 2개: 청크 9칸 / 공고명만.
방법: 키워드(BM25) 단독, 임베딩 학습 전/후 단독, 임베딩 학습 전/후 + 키워드(하이브리드), LLM 5개, Decision Model flash.
순위: 점수 방법은 eval_v3.rank()(동점 가운데 순위), LLM은 응답 목록 위치(반복 번호는 첫 등장, 30위 밖 = 없음), Decision Model은 그 세션의 쌍별 순위.
1단계(임베딩 환경): <_embedding_v3_env python> -I -X utf8 runs/044/method_compare_xlsx_harness.py calc  → runs/044/method_compare_harness.json
2단계(.venv):       .venv/Scripts/python.exe -X utf8 runs/044/method_compare_xlsx_harness.py xlsx  → runs/044/방법별_성능비교_harness.xlsx
공고명 입력의 키워드 단독·하이브리드는 11-23에 측정(공고명 Kiwi 토큰 = name_enc_harness/kiwi_tokens_name_harness.json)."""
import json, sys, os, hashlib
from collections import defaultdict
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
MODE = sys.argv[1]
EXP = Path(__file__).resolve().parents[2]; H = EXP / 'runs/044'; JS = H / 'method_compare_harness.json'
K8 = 8
# LLM 정가(캐시 없음, claude-api 스킬 단가표 2026-11-03 x 토큰): VERIFY_LOG_harness.md 2026-11-12T18:14:19 검증 표에서 옮김
LLM = [('claude-haiku-5-5', 'low', 'Haiku 5.5 low', {'chunk': 0.57, 'name': 0.09}), ('claude-sonnet-5-5', 'low', 'Sonnet 5.5 low', {'chunk': 11.37, 'name': 1.37}),
       ('claude-sonnet-5-5', 'high', 'Sonnet 5.5 high', {'chunk': 11.81, 'name': 1.78}), ('claude-opus-5-5', 'low', 'Opus 5.5 low', {'chunk': 23.41, 'name': 3.41}),
       ('claude-opus-5-5', 'high', 'Opus 5.5 high', {'chunk': 23.54, 'name': 3.50})]

if MODE == 'calc':
    import numpy as np
    EVF = H / 'pool668_newreview_harness/eval_set_pool668.json'
    src = (EXP / 'eval_v3.py').read_text(encoding='utf-8'); old = "EVAL = HERE / 'runs' / '009' / 'eval_set_v2.json'"; assert src.count(old) == 1
    src = src.replace(old, f"EVAL = Path({str(EVF)!r})")
    sys.argv = ['eval_v3.py', '044/pool668_newreview_harness', '30']; g = {'__file__': str(EXP / 'eval_v3.py'), '__name__': 'core'}; exec(src[:src.index('cache = {}')], g)
    R, ix, ev, rank, N = g['R'], g['ix'], g['ev'], g['rank'], g['N']
    T = defaultdict(set)
    for gr, a, b in ev['pairs']: T[gr] |= {ix[a], ix[b]}
    QS = sorted((gr, q) for gr, m in T.items() for q in m); assert len(QS) == 35 and len(T) == 14
    mm0 = g['mm'].copy()
    def emb_chunk(p):
        with np.load(p) as z: e = dict(zip(z['h'].tolist(), z['v']))
        return np.stack([e[r['sha256']] for r in R]).astype(np.float32)
    C = [json.loads(l) for l in open(H / 'pool668_newreview_harness/chunks.jsonl', encoding='utf-8')]; assert len(C) == N
    def emb_name(p):
        with np.load(p) as z: e = dict(zip(z['h'].tolist(), z['v']))
        return np.stack([e[hashlib.sha256(c['name'].encode()).hexdigest()] for c in C]).astype(np.float32)
    def rowmm(X):
        S = X @ X.T; np.fill_diagonal(S, np.inf); lo = S.min(1); np.fill_diagonal(S, -np.inf); return lo, S.max(1)
    def score_ranks(X, kind):
        g['E'] = X; mm = mm0.copy(); mm[:, 2], mm[:, 3] = rowmm(X); g['mm'] = mm; out = []
        for gr, q in QS:
            sc = np.array(g['rows_of'](q)[kind], float); sc[q] = np.nan
            out.append([rank(sc, q, o) for o in sorted(T[gr] - {q})])
        return out
    def cos_ranks(X):
        S = X @ X.T; out = []
        for gr, q in QS:
            sc = S[q].astype(float).copy(); sc[q] = np.nan
            out.append([rank(sc, q, o) for o in sorted(T[gr] - {q})])
        return out
    base = emb_chunk(EXP / 'runs/035/emb_cache/qwen_chunk.npz'); ft = emb_chunk(H / 'seed_rep_harness/qwen668_s42.npz')
    EP = {lab: H / f'stab40_harness/{r}' for lab, r in (('1 epoch', 'qwen_cropB3batch128_1ep_s42_20261106_night3'), ('2 epoch', 'qwen_cropB3batch128_2ep_s42_20261118_stab40'))}
    def emb_file(p, kf):
        with np.load(p) as z: e = dict(zip(z['h'].tolist(), z['v']))
        return np.stack([e[kf(c)] for c in C]).astype(np.float32)
    nb = emb_name(H / 'name_enc_harness/qwen_name_base.npz'); nf = emb_name(H / 'name_enc_harness/qwen_name_b3b128.npz')
    # 11-23 공고명만 입력의 키워드·하이브리드: eval_v3.py의 BM25 블록(같은 식·같은 상수)을 공고명 Kiwi 토큰으로 다시 실행. mm은 knn_build.py block()과 같이 행 전체 min/max
    NT = json.load(open(H / 'name_enc_harness/kiwi_tokens_name_harness.json', encoding='utf-8')); blk = src[src.index('vocab, rows, cols, vals'):src.index('cache = {}')]
    def bm_space(toks):
        g2 = dict(g); g2['toks'] = toks; g2['mm'] = np.zeros((N, 4)); exec(blk, g2)
        V_ = np.stack([g2['rows_of'](q)['VN'] for q in range(N)]); return g2, np.nanmin(V_, 1), np.nanmax(V_, 1)
    gc, clo, chi = bm_space(g['toks']); assert np.allclose(clo, mm0[:, 0]) and np.allclose(chi, mm0[:, 1]), '청크 토큰으로 knn mm 재현 실패'
    gn, nlo, nhi = bm_space([NT[hashlib.sha256(c['name'].encode()).hexdigest()] for c in C])
    def name_ranks(X, kind):
        gn['E'] = X; mm = np.zeros((N, 4)); mm[:, 0], mm[:, 1] = nlo, nhi; mm[:, 2], mm[:, 3] = rowmm(X); gn['mm'] = mm; out = []
        for gr, q in QS:
            sc = np.array(gn['rows_of'](q)[kind], float); sc[q] = np.nan
            out.append([rank(sc, q, o) for o in sorted(T[gr] - {q})])
        return out
    def llm(model, eff, inp, sub=''):
        rr = []
        for i, (gr, q) in enumerate(QS):
            f = json.load(open(H / f'llm_rank_harness/{model}_{eff}_{inp}{sub}/q{i:02d}.json', encoding='utf-8')); assert f['query'] == q
            rr.append([f['mate_rank'][str(m)] for m in sorted(T[gr] - {q})])
        return rr
    def dm(f):
        d = json.load(open(H / 'decision_model_harness' / f, encoding='utf-8')); dr = {}
        for p in d['pairs']: dr[(ix[p['a']], ix[p['b']])] = p['rank_b_from_a']; dr[(ix[p['b']], ix[p['a']])] = p['rank_a_from_b']
        return [[dr[(q, m)] for m in sorted(T[gr] - {q})] for gr, q in QS]
    M = {'chunk': [], 'name': []}
    M['chunk'] += [('키워드', '키워드(BM25) 단독', score_ranks(base, 'VN'), None),
                   ('임베딩', '임베딩 학습 전 단독', score_ranks(base, 'Qwen'), None),
                   ('하이브리드', '임베딩 학습 전 + 키워드', score_ranks(base, 'VN+Qwen'), None),
                   ('임베딩', '임베딩 학습 후(5스텝 seed 42) 단독', score_ranks(ft, 'Qwen'), None),
                   ('하이브리드', '임베딩 학습 후(5스텝 seed 42) + 키워드', score_ranks(ft, 'VN+Qwen'), None)]
    EPN = []
    for lab, d in EP.items():
        X = emb_file(d / 'q668.npz', lambda c: hashlib.sha256(c['text'].encode()).hexdigest())
        M['chunk'] += [('임베딩', f'임베딩 학습 후({lab} seed 42) 단독', score_ranks(X, 'Qwen'), None), ('하이브리드', f'임베딩 학습 후({lab} seed 42) + 키워드', score_ranks(X, 'VN+Qwen'), None)]
        EPN += [('임베딩', f'임베딩 학습 후({lab} seed 42) 단독', cos_ranks(emb_file(d / 'name.npz', lambda c: hashlib.sha256(c['name'].encode()).hexdigest())), None)]
    M['name'] += [('키워드', '키워드(BM25) 단독', name_ranks(nb, 'VN'), None), ('임베딩', '임베딩 학습 전 단독', cos_ranks(nb), None),
                  ('하이브리드', '임베딩 학습 전 + 키워드', name_ranks(nb, 'VN+Qwen'), None), ('임베딩', '임베딩 학습 후(5스텝 seed 42) 단독', cos_ranks(nf), None),
                  ('하이브리드', '임베딩 학습 후(5스텝 seed 42) + 키워드', name_ranks(nf, 'VN+Qwen'), None)]
    M['name'] += EPN  # 1·2 epoch는 5스텝 다음에
    for inp in ('chunk', 'name'):
        for model, eff, lab, cost in LLM:
            if (model, eff, inp) == ('claude-haiku-5-5', 'low', 'chunk'):  # 11-19 재실행: 1차(11-12)와 2차를 모두 보고
                M[inp].append(('LLM', f'LLM {lab} (1차 11-12)', llm(model, eff, inp, '__run1_20261112'), cost[inp]))
                M[inp].append(('LLM', f'LLM {lab} (2차 11-19)', llm(model, eff, inp), None))
                for e2 in ('low', 'high'):  # 11-19: 100위까지 답하게 한 실행(완료된 것만)
                    if len(list((H / f'llm_rank_harness/{model}_{e2}_{inp}_top100').glob('q??.json'))) == 35:
                        M[inp].append(('LLM', f'LLM Haiku 5.5 {e2} (상위 100개 응답)', llm(model, e2, inp, '_top100'), None))
                continue
            M[inp].append(('LLM', f'LLM {lab}', llm(model, eff, inp), cost[inp]))
    M['chunk'].append(('Decision Model', 'Decision Model flash (pplx-decider-flash-v1.2)', dm('decider_eval_pool668_full9_harness.json'), None))
    M['name'].append(('Decision Model', 'Decision Model flash (pplx-decider-flash-v1.2)', dm('decider_eval_pool668_harness.json'), None))
    M['name'].append(('Decision Model', 'Decision Model 27B (pplx-decider-v1.2)', dm('decider_eval_pool668_27b_harness.json'), None))
    M['chunk'].append(('Decision Model', 'Decision Model 27B (pplx-decider-v1.2)', dm('decider_eval_pool668_27b_full9_harness.json'), None))
    out = {'K': K8, 'queries': [{'group': gr, 'row': C[q]['id']['sheet_row'], 'mates': [C[m]['id']['sheet_row'] for m in sorted(T[gr] - {q})]} for gr, q in QS], 'inputs': {}}
    for inp, rows in M.items():
        res = []
        for kind, lab, rs, cost in rows:
            if rs is None: res.append({'kind': kind, 'method': lab, 'measured': False}); continue
            last = [None if any(r is None for r in x) else float(max(x)) for x in rs]
            rec = [float(np.mean([(r is not None and r <= K8) for r in x])) for x in rs]
            known = [v for v in last if v is not None]
            res.append({'kind': kind, 'method': lab, 'measured': True, 'worst': None if None in last else max(last), 'beyond30': sum(v is None for v in last),
                        'recall@8': float(np.mean(rec)), 'all_within_8': sum(all(r is not None and r <= K8 for r in x) for x in rs),
                        'mean_last': None if None in last else float(np.mean(last)), 'median_last': float(np.median([1e9 if v is None else v for v in last])) if sum(v is None for v in last) < len(last) / 2 else None,
                        'MAP': None if None in last else float(np.mean([np.mean([(k + 1) / r for k, r in enumerate(sorted(x))]) for x in rs])), 'mate_ties': sum(len(set(x)) < len(x) for x in rs if None not in x),
                        'mean_last_known': float(np.mean(known)), 'median_last_known': float(np.median(known)), 'last': last, 'mate_ranks': rs, 'cost_usd': cost})
            print(inp, f'{lab:44s}', res[-1]['worst'], round(res[-1]['recall@8'], 3), res[-1]['all_within_8'])
        out['inputs'][inp] = res
    tmp = JS.with_suffix('.tmp'); tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8'); os.replace(tmp, JS)

elif MODE == 'xlsx':
    from openpyxl import Workbook
    from openpyxl.styles import Font, PatternFill, Alignment
    D = json.load(open(JS, encoding='utf-8'))
    C = {json.loads(l)['id']['sheet_row']: json.loads(l)['name'] for l in open(H / 'pool668_newreview_harness/chunks.jsonl', encoding='utf-8')}
    B = Font(name='Arial', bold=True); NF = Font(name='Arial'); BEST = PatternFill('solid', fgColor='E2EFDA'); GREY = PatternFill('solid', fgColor='EDEDED')
    SRC = 'runs/044/method_compare_harness.json'; INP = {'chunk': '청크 9칸', 'name': '공고명만'}
    wb = Workbook(); wb.remove(wb.active)
    def fmt(ws, widths, freeze):
        for row in ws.iter_rows():
            for c in row:
                if not c.font.b: c.font = NF
        for i, w in enumerate(widths): ws.column_dimensions[chr(65 + i)].width = w
        ws.freeze_panes = freeze
    # 1 요약: 입력 조건마다 방법 12행
    ws = wb.create_sheet('요약')
    hdr = ['입력', '방법 종류', '방법', '최악 마지막 정답 순위', '평균 마지막 정답 순위', '중앙값 마지막 정답 순위', 'Recall@8 (macro)', 'MAP', '정답 전부 8위 안 질의 수(/35)', '응답 범위 밖 질의 수(30위 또는 100위)', 'LLM 정가 비용(USD, 35질의)', '비고', '출처 파일']
    ws.append(hdr)
    for c in ws[1]: c.font = B; c.alignment = Alignment(wrap_text=True, vertical='top')
    for inp, res in D['inputs'].items():
        meas = [r for r in res if r['measured'] and r['worst'] is not None]; bw = min(r['worst'] for r in meas); br = max(r['recall@8'] for r in res if r['measured']); bm = max(r['MAP'] for r in meas)
        for r in res:
            if not r['measured']:
                ws.append([INP[inp], r['kind'], r['method'], '미측정', '', '', '', '', '', '', '', '공고명 입력으로는 실행하지 않음', SRC])
                for c in ws[ws.max_row]: c.fill = GREY
                continue
            notes = []
            if r['beyond30']: notes.append(('100위' if '100' in r['method'] else '30위') + ' 밖 질의가 있어 최악·평균·MAP 계산 불가(중앙값은 밖인 질의를 맨 뒤로 두고 계산)')
            if r['mate_ties']: notes.append(f"정답끼리 동점 질의 {r['mate_ties']}개(MAP이 표준 정의와 조금 다름)")
            na = '계산 불가'
            ws.append([INP[inp], r['kind'], r['method'], r['worst'] if r['worst'] is not None else '30위 밖 있음', na if r['mean_last'] is None else round(r['mean_last'], 2),
                       na if r['median_last'] is None else r['median_last'], round(r['recall@8'], 3), na if r['MAP'] is None else round(r['MAP'], 4), r['all_within_8'], r['beyond30'],
                       r['cost_usd'] if r['cost_usd'] is not None else '', '; '.join(notes), SRC])
            row = ws[ws.max_row]
            if r['worst'] == bw: row[3].fill = BEST
            if r['recall@8'] == br: row[6].fill = BEST
            if r['MAP'] is not None and r['MAP'] == bm: row[7].fill = BEST
    ws.append([]); ws.append(['풀: 2026 공고 668개(질의 자신 제외 후보 667개), 평가셋 그룹 14개·질의 35개(경영). 초록 칸 = 입력 조건 안에서 가장 좋은 값.'])
    ws.append(['마지막 정답 순위 = 질의마다 정답 중 가장 뒤에 있는 정답의 순위. 최악·평균·중앙값은 질의 35개에 대해 계산. MAP = 질의마다 정답 순위를 오름차순 정렬해 k번째 정답의 k/순위를 평균(AP)한 뒤 35개 평균.'])
    ws.append(['임베딩 점수는 저장된 벡터의 내적(벡터 길이 0.996~1.004, bf16 반올림). 임베딩 학습 후 조건은 seed 42 한 번씩.'])
    ws.append(['Recall@8 = 질의마다 (정답 후보 중 8위 안에 든 비율)의 평균. LLM은 상위 30개만 답하므로 30위 밖은 순위 없음. 비용은 토큰 × 정가(캐시 할인 없음), 임베딩·키워드·Decision Model은 로컬 GPU/CPU라 비워 둠.'])
    fmt(ws, [9, 13, 44, 12, 12, 12, 11, 9, 13, 10, 14, 40, 32], 'D2')
    # 2·3 질의별 마지막 정답 순위 (입력 조건마다 시트 하나)
    for inp, res in D['inputs'].items():
        ws = wb.create_sheet(f'질의별_{INP[inp]}')
        meas = [r for r in res if r['measured']]
        ws.append(['질의', '그룹', '질의 공고명', '정답 공고명'] + [r['method'] for r in meas] + ['출처 파일', '질의 마스터 행', '정답 마스터 행'])
        for c in ws[1]: c.font = B; c.alignment = Alignment(wrap_text=True, vertical='top')
        for i, q in enumerate(D['queries']):
            vals = [r['last'][i] for r in meas]
            ws.append([f'Q{i + 1:02d}', q['group'], C[q['row']], ', '.join(C[m] for m in q['mates'])] + [v if v is not None else '30위 밖' for v in vals]
                      + [SRC, q['row'], ', '.join(map(str, q['mates']))])
            nums = [v for v in vals if v is not None]
            if nums:
                for j, v in enumerate(vals):
                    if v == min(nums): ws.cell(ws.max_row, 5 + j).fill = BEST
        ws.append(['최악', '', '', ''] + [r['worst'] if r['worst'] is not None else '30위 밖 있음' for r in meas]);
        for c in ws[ws.max_row]: c.font = B
        fmt(ws, [7, 7, 40, 50] + [14] * len(meas) + [30, 10, 14], 'E2')
    out = H / '방법별_성능비교_harness.xlsx'; tmp = H / '방법별_성능비교_harness.tmp.xlsx'; wb.save(tmp); os.replace(tmp, out); print(out)
