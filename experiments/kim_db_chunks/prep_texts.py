"""Freeze every chunk text used by the retrieval pipeline into one file (runs/<run>/texts.json).

Types: name, all, purpose_content, beneficiary, legal (Kim, from chunks_export.jsonl) and nam_all, nam_content
(eval-group programs get 남윤형 엑셀 행 text, every other program keeps its Kim 'all' chunk).
nam_all  = every 사업목록 column except budget columns (사용자 요청 2026-10-28: "예산 빼고 다 합쳐서").
nam_content = program-description columns only (control for leakage from group id/name, 효율화방안, mapping columns).
Usage: .venv/Scripts/python.exe -I -X utf8 prep_texts.py <run>
"""
import hashlib
import json
import sys
from pathlib import Path

import openpyxl

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs' / sys.argv[1]
EXPORT = RUN / 'chunks_export.jsonl'
NAM = Path('<DEV_EVAL_ROOT>/jh_dev/govparse-claude-lab/outputs/goldenset_similar_groups_xlsx/유사사업그룹_표추출.xlsx')
NAM_SHA = 'f185d861455f2e9fee9fe35dd517452cdaf342d653ec7d7ffa7432172f32aee2'
NAM_BUDGET_COLS = {'26예산(억원)', '예산서_2026금액(백만원)', '원문대조_금액'}
NAM_CONTENT_COLS = ['사업명', '소관부처', '사업목적', '지원대상(업종)', '지원내용', '지원조건', '지원방식', '특이사항']
assert hashlib.sha256(NAM.read_bytes()).hexdigest() == NAM_SHA

progs = [json.loads(x) for x in EXPORT.read_text(encoding='utf-8').splitlines()]
ev = json.loads((RUN / 'eval_set_exact_2026.json').read_text(encoding='utf-8'))
eval_names = {m['keys'][0]: m['사업명'] for m in ev['matched_programs'] if m['그룹ID'] in ev['groups_evaluable']}
it = openpyxl.load_workbook(NAM, read_only=True)['사업목록'].iter_rows(values_only=True)
h = list(next(it))
nam = {}
for r in it:
    d = dict(zip(h, r))
    for k, nm in eval_names.items():
        if d['사업명'] == nm:
            nam[k] = d
assert set(nam) == set(eval_names)
merged = lambda d: {x.strip() for x in str(d.get('병합(공통)항목') or '').replace('，', ',').split(',') if x.strip()}  # noqa: E731
fmt = lambda d, cols: ' / '.join(f'{c}: {d[c]}' for c in cols if d.get(c) not in (None, ''))  # noqa: E731
over = {'nam_all': {k: fmt(d, [c for c in h if c not in NAM_BUDGET_COLS]) for k, d in nam.items()},
        'nam_content': {k: fmt(d, NAM_CONTENT_COLS) for k, d in nam.items()},
        # nam_content_nomerge: also drop columns the table merged across the group (병합(공통)항목 lists them; the same
        # cell text was copied to every program of the group, e.g. G19 사업목적) -> the strictest leakage control
        'nam_content_nomerge': {k: fmt(d, [c for c in NAM_CONTENT_COLS if c == '사업명' or c not in merged(d)]) for k, d in nam.items()}}
texts = {'name': {p['key']: p['name'] for p in progs}}
for t in ('all', 'purpose_content', 'beneficiary', 'legal'):
    texts[t] = {p['key']: p[t] for p in progs}
for t, o in over.items():
    texts[t] = {p['key']: o.get(p['key'], p['all']) for p in progs}
has = {t: {p['key']: bool(p['has'][t]) for p in progs} for t in ('purpose_content', 'beneficiary', 'legal')}
out = {'keys': [p['key'] for p in progs], 'names': {p['key']: p['name'] for p in progs}, 'texts': texts, 'has': has,
       'eval_groups': {g: [m['keys'][0] for m in ev['matched_programs'] if m['그룹ID'] == g] for g in ev['groups_evaluable']},
       'export_sha256': hashlib.sha256(EXPORT.read_bytes()).hexdigest(), 'nam_sha256': NAM_SHA,
       'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(RUN / 'texts.json').write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
print({t: len(v) for t, v in texts.items()}, hashlib.sha256((RUN / 'texts.json').read_bytes()).hexdigest())
for k in nam:
    print(k, '| merged:', sorted(merged(nam[k])), '| nomerge:', over['nam_content_nomerge'][k][:100])
