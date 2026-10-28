"""Build per-program chunks for the chunking experiment from Kim's tabular data (chunk_config.json).

Program unit = l2 key of a name-column value (same normalization as merge_program_names.py), 2026, central only.
For every row (2026) of every central sheet/table that has a name column, the row's texts are attached to each name in it:
  all              = every text cell of the row except the name columns (condition A, one chunk)
  purpose_content  = configured purpose/content columns            (condition B, chunk 1)
  beneficiary      = configured beneficiary columns                (condition B, chunk 2)
Texts are deduplicated per program (copies of the same table in several files add nothing). Read-only on sources.
Usage: .venv/Scripts/python.exe -I -X utf8 build_chunks.py <run_name>   -> runs/<run>/chunks.duckdb
"""
import ast
import csv
import hashlib
import json
import re
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import duckdb
import openpyxl

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(2**31 - 1)
HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs' / sys.argv[1]
DB = RUN / 'chunks.duckdb'
if DB.exists():
    raise SystemExit(f'{DB} exists')
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
CC = json.loads((HERE / 'chunk_config.json').read_text(encoding='utf-8'))
SRC = HERE / 'merge_program_names.py'
ns = {'re': re, 'CFG': CFG, 'NORM': CFG['normalize'], 'YCFG': CFG['year'],
      'L1_RE': re.compile(CFG['normalize']['L1_remove']), 'L2_RES': [re.compile(p) for p in CFG['normalize']['L2_strip_patterns']],
      'YEAR_RE': re.compile(CFG['year']['name_pattern'])}
exec(compile(ast.Module(body=[n for n in ast.parse(SRC.read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef)
                              and n.name in {'l1', 'l2', 'head', 'year_col', 'year_of_text', 'year_of_value'}], type_ignores=[]),
             str(SRC), 'exec'), ns)
l1, l2, head, year_col, year_of_text, year_of_value = (ns[k] for k in ('l1', 'l2', 'head', 'year_col', 'year_of_text', 'year_of_value'))
COL2LEVEL = {c: lv for lv, cols in CFG['name_columns'].items() if not lv.startswith('_') for c in cols}
FIELD_SETS = {t: set(cols) for t, cols in CC['fields'].items()}  # purpose_content, beneficiary, legal
IDC = set(CC['id_columns'])
YEAR = CC['year']
NUMLIKE = re.compile(r'^[\d\s,.\-:/%()△]+$')

KINDS = ['all'] + list(CC['fields'])
prog = defaultdict(lambda: {'names': set(), 'sources': set(), **{t: {} for t in KINDS}})
stats = defaultdict(int)


def add_row(header, row, unit_label, fy, linked_names=None):
    """linked_names: names reached through 사업_id for tables without a name column (year comes from the id's row)"""
    hs = [head(h) for h in header]
    if linked_names is None:
        yi, _ = year_col(header)
        ry = (year_of_value(row[yi]) if yi is not None and yi < len(row) else None) or fy
        if ry != YEAR:
            return
        names = [str(row[i]).strip() for i, h in enumerate(hs) if h in COL2LEVEL and i < len(row) and row[i] is not None and str(row[i]).strip()]
    else:
        names = linked_names
    if not names:
        return
    texts = {t: [] for t in KINDS}
    for i, h in enumerate(hs):
        v = row[i] if i < len(row) else None
        if v is None or h in COL2LEVEL or h in IDC or not isinstance(v, str):
            continue
        v = v.strip()
        if len(v) < CC['all_min_chars'] or NUMLIKE.match(v):
            continue
        texts['all'].append((h, v))
        for t, cols in FIELD_SETS.items():
            if h in cols:
                texts[t].append((h, v))
    stats['rows_linked' if linked_names is not None else 'rows_2026'] += 1
    for nm in names:
        k = l2(nm)
        if len(k) < CFG['min_name_len']:
            continue
        p = prog[k]
        p['names'].add(nm)
        p['sources'].add(unit_label)
        for t, items in texts.items():
            for h, v in items:
                p[t].setdefault(v, h)  # dedup by text, keep first column label


files = json.loads((HERE / 'runs' / '003' / 'files.json').read_text(encoding='utf-8'))
for f in files:
    path = f['path']
    if f['is_copy'] and CC['skip_copies'] or any(m in path for m in CC['exclude_path_markers']) or f['error']:
        continue
    units = [u['unit'] for u in f['units'] if u['status'] == 'ok' and not any(m in u['unit'] for m in CC['exclude_unit_markers'])]
    if not units:
        continue
    p = Path(path)
    ext = p.suffix.lower()
    print(path, len(units), flush=True)
    if ext in ('.sqlite', '.db'):
        con = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        id2names = defaultdict(set)  # 사업_id -> 2026 names, from tables that have both
        for t in units:
            header = [r[1] for r in con.execute(f'pragma table_info("{t}")')]
            fy = year_of_text(t) or year_of_text(p.name)
            hs = [head(h) for h in header]
            idi = next((i for i, h in enumerate(hs) if h in IDC), None)
            yi, _ = year_col(header)
            for row in con.execute(f'select * from "{t}"'):
                add_row(header, row, f'{p.name}:{t}', fy)
                if idi is not None and row[idi] is not None:
                    ry = (year_of_value(row[yi]) if yi is not None else None) or fy
                    if ry == YEAR:
                        id2names[row[idi]].update(str(row[i]).strip() for i, h in enumerate(hs)
                                                  if h in COL2LEVEL and row[i] is not None and str(row[i]).strip())
        if id2names:
            for (t,) in con.execute("select name from sqlite_master where type='table'").fetchall():
                header = [r[1] for r in con.execute(f'pragma table_info("{t}")')]
                hs = [head(h) for h in header]
                idi = next((i for i, h in enumerate(hs) if h in IDC), None)
                if t in units or idi is None or not any(h in c for h in hs for c in FIELD_SETS.values()):
                    continue  # only name-less tables that carry a configured field
                for row in con.execute(f'select * from "{t}"'):
                    nm = id2names.get(row[idi])
                    if nm:
                        add_row(header, row, f'{p.name}:{t}(사업_id)', None, linked_names=sorted(nm))
        con.close()
    elif ext in ('.xlsx', '.xlsm'):
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        for ws in wb.worksheets:
            if ws.title not in units:
                continue
            rows = ws.iter_rows(values_only=True)
            header = None
            for _ in range(CFG['header_scan_rows']):
                r = next(rows, None)
                if r is None:
                    break
                if any(head(x) in COL2LEVEL for x in r):
                    header = [str(x) if x is not None else '' for x in r]
                    break
            if header is None:
                continue
            fy = year_of_text(ws.title) or year_of_text(p.name)
            for r in rows:
                add_row(header, r, f'{p.name}:{ws.title}', fy)
        wb.close()
    elif ext == '.csv':
        for enc in CFG['csv_encodings']:
            try:
                with open(path, encoding=enc, newline='') as fh:
                    rd = csv.reader(fh)
                    header = next(rd, [])
                    fy = year_of_text(p.name)
                    for r in rd:
                        add_row(header, r, p.name, fy)
                break
            except UnicodeDecodeError:
                continue


def chunk(k, p, kind):
    name = sorted(p['names'], key=lambda s: (-len(s), s))[0]
    body = ' / '.join(f'{h}: {v}' for v, h in p[kind].items())
    return (name + ('\n' + body if body else ''))[:CC['max_chunk_chars']]


con = duckdb.connect(str(DB))
CONDITIONS = {'A_all': ['all'], 'B_split': ['purpose_content', 'beneficiary'], 'C_split_legal': ['purpose_content', 'beneficiary', 'legal']}
con.execute('create table programs(key_l2 varchar, name varchar, n_names int, n_sources int, sources varchar[], '
            + ', '.join(f'len_{t} int' for t in KINDS) + ')')
con.execute('create table chunks(key_l2 varchar, condition varchar, chunk_type varchar, text varchar)')
prow, crow = [], []
for k, p in prog.items():
    name = sorted(p['names'], key=lambda s: (-len(s), s))[0]
    prow.append([k, name, len(p['names']), len(p['sources']), sorted(p['sources'])] + [len(' '.join(p[t])) for t in KINDS])
    for cond, types in CONDITIONS.items():
        for t in types:
            crow.append([k, cond, t, chunk(k, p, t)])
con.executemany(f'insert into programs values ({",".join("?" * (5 + len(KINDS)))})', prow)
con.executemany('insert into chunks values (?,?,?,?)', crow)
meta = {'rows_2026': stats['rows_2026'], 'rows_linked_by_id': stats['rows_linked'], 'programs': len(prog),
        **{f'with_{t}': sum(1 for p in prog.values() if p[t]) for t in KINDS}, 'conditions': CONDITIONS,
        'chunk_config_sha256': hashlib.sha256((HERE / 'chunk_config.json').read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
con.execute('create table meta as select ? as json', [json.dumps(meta, ensure_ascii=False)])
con.close()
print(json.dumps(meta, ensure_ascii=False, indent=1))
