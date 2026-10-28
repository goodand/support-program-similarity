"""남윤형 박사님 평가셋(경영 분야) 사업명을 김미정 자료 병합 DB(2025)와 대응 (독립 실험 1단계 후속).

Evaluation file: 유사사업그룹_표추출.xlsx, sheet 사업목록, 분야 startswith '경영' (사용자: 경영 테이블 21개).
Matching key = l2 from merge_program_names.py (특수문자·띄어쓰기·괄호 무시), applied to 사업명 and 사업명_본명.
Read-only on the DuckDB. Usage: .venv python -I -X utf8 match_eval_names.py <run_name>  -> runs/<run>/eval_match.json
"""
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

import duckdb
import openpyxl

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs' / sys.argv[1]
EVAL = Path(r'<DEV_EVAL_ROOT>\jh_dev\govparse-claude-lab\outputs\goldenset_similar_groups_xlsx\유사사업그룹_표추출.xlsx')
SRC = HERE / 'merge_program_names.py'
NORM = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))['normalize']
ns = {'re': re, 'NORM': NORM, 'L1_RE': re.compile(NORM['L1_remove']), 'L2_RES': [re.compile(p) for p in NORM['L2_strip_patterns']]}
exec(compile(ast.Module(body=[n for n in ast.parse(SRC.read_text(encoding='utf-8')).body
                              if isinstance(n, ast.FunctionDef) and n.name in {'l1', 'l2'}], type_ignores=[]), str(SRC), 'exec'), ns)
l2 = ns['l2']
LOCAL = ("file like '%businfo%' or file like '%지자체사업정보DB%' or file like '%세출예산%'")

ws = openpyxl.load_workbook(EVAL, read_only=True)['사업목록']
rows = ws.iter_rows(values_only=True)
h = list(next(rows))
recs = [dict(zip(h, r)) for r in rows if r[0]]
mgmt = [r for r in recs if str(r['분야']).startswith('경영')]

con = duckdb.connect(str(RUN / 'kim_programs.duckdb'), read_only=True)
out = []
for r in mgmt:
    keys = sorted({l2(r['사업명']), l2(r['사업명_본명'] or r['사업명'])})
    hit = con.execute(f"""select key_l2, case when {LOCAL} then '지자체' else '중앙' end g, count(distinct file) nf,
                          list(distinct name)[:5] names, list(distinct replace(file,'<DEV_EVAL_ROOT>\\',''))[:5] files
                          from names where key_l2 in ({','.join('?' * len(keys))}) group by 1, 2""", keys).fetchall()
    gs = {g for _, g, *_ in hit}
    out.append({'그룹ID': r['그룹ID'], '분야': r['분야'], '사업명': r['사업명'], '사업명_본명': r['사업명_본명'], '소관부처': r['소관부처'],
                'keys': keys, 'status': '중앙 일치' if '중앙' in gs else ('지자체만 일치' if gs else '불일치'),
                'hits': [{'key': k, 'group': g, 'n_files': nf, 'names': nm, 'files': fs} for k, g, nf, nm, fs in hit]})
summary = {s: sum(o['status'] == s for o in out) for s in ('중앙 일치', '지자체만 일치', '불일치')}
summary |= {'eval_rows': len(mgmt), 'eval_groups': len({o['그룹ID'] for o in out}),
            'eval_sha256': hashlib.sha256(EVAL.read_bytes()).hexdigest(), 'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(RUN / 'eval_match.json').write_text(json.dumps({'summary': summary, 'rows': out}, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(summary, ensure_ascii=False, indent=1))
for o in out:
    if o['status'] != '중앙 일치':
        print(o['status'], '|', o['그룹ID'], o['사업명'], '|', o['소관부처'], '|', o['keys'])
