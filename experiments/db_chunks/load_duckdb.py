"""Load a merge_program_names.py run into one DuckDB file (사용자 결정 2026-10-28: DuckDB로 병합).

Normalization (l1, l2) is NOT re-implemented: the function defs are taken from merge_program_names.py via AST and
registered as DuckDB UDFs, so the DB keys equal the run's keys. Asserts the DB counts match counts.json.
Usage: .venv/Scripts/python.exe -I -X utf8 load_duckdb.py <run_name>   -> runs/<run_name>/programs.duckdb
"""
import ast
import json
import re
import sys
from pathlib import Path

import duckdb

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs' / sys.argv[1]
DB = RUN / 'programs.duckdb'
if DB.exists():
    raise SystemExit(f'{DB} exists')
SRC = HERE / 'merge_program_names.py'
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
NORM = CFG['normalize']
ns = {'re': re,  # l2 calls l1, both taken from the source below
 'NORM': NORM, 'L1_RE': re.compile(NORM['L1_remove']), 'L2_RES': [re.compile(p) for p in NORM['L2_strip_patterns']]}
tree = ast.parse(SRC.read_text(encoding='utf-8'))
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {'l1', 'l2'}], type_ignores=[]),
             str(SRC), 'exec'), ns)

con = duckdb.connect(str(DB))
con.create_function('l1', ns['l1'], ['VARCHAR'], 'VARCHAR')
con.create_function('l2', ns['l2'], ['VARCHAR'], 'VARCHAR')
files = json.loads((RUN / 'files.json').read_text(encoding='utf-8'))
con.execute('create table files(path varchar, size bigint, mtime_ns bigint, sha256 varchar, error varchar, is_copy boolean)')
con.executemany('insert into files values (?,?,?,?,?,?)', [[f['path'], f['size'], f['mtime_ns'], f['sha256'], f['error'], f['is_copy']] for f in files])
con.execute('create table units(path varchar, unit varchar, status varchar, distinct_names bigint, "rows" bigint)')
con.executemany('insert into units values (?,?,?,?,?)', [[f['path'], u['unit'], u['status'], u['distinct_names'], u['rows']] for f in files for u in f['units']])
con.execute(f"""create table names_long as
  select name, level, "column", year, year_source, count, file, unit, is_copy, l1(name) as key_l1, l2(name) as key_l2
  from read_json('{(RUN / 'names_long.jsonl').as_posix()}', format='newline_delimited',
                 columns={{name:'VARCHAR', level:'VARCHAR', "column":'VARCHAR', year:'INTEGER', year_source:'VARCHAR', count:'BIGINT',
                          file:'VARCHAR', unit:'VARCHAR', is_copy:'BOOLEAN'}})""")
con.execute(f"create view names as select * from names_long where year = {CFG['year']['target']} and length(key_l1) >= {CFG['min_name_len']}")
con.execute("""create table programs as
  select key_l2, arg_max(name, count) as name, count(distinct name) as n_variants, count(distinct file) as n_files,
         sum(count) as n_rows, list(distinct level order by level) as levels
  from names group by key_l2""")

# integrity vs the run's own counts
counts = json.loads((RUN / 'counts.json').read_text(encoding='utf-8'))
assert con.execute('select sum(count) from names_long').fetchone()[0] == counts['name_rows']
for scope, where in (('전체', ''), ('사본 제외', 'where not is_copy')):
    got = con.execute(f'select count(distinct name), count(distinct key_l1), count(distinct key_l2) from names {where}').fetchone()
    exp = counts[scope]['전체 수준']
    assert list(got) == [exp['L0'], exp['L1'], exp['L2']], (scope, got, exp)
print('ok', DB, con.execute('select count(*) from programs').fetchone()[0], 'programs (L2)')
con.close()
