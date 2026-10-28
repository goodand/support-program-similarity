"""자료 제공자 A 자료 전체에서 사업명을 모아 병합·중복 제거·개수 집계 (독립 실험 1단계).

Reads config.json. Read-only on every source (sqlite opened with mode=ro). Per-file results are cached in cache/
keyed by (path, size, mtime), so an interrupted run resumes. Output: runs/<name>/ (refuses a non-empty folder).
Usage: python -I -X utf8 merge_program_names.py <run_name>
"""
import csv
import hashlib
import json
import os
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

sys.stdout.reconfigure(encoding='utf-8')
csv.field_size_limit(2**31 - 1)
HERE = Path(__file__).resolve().parent
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))
OUT = HERE / 'runs' / sys.argv[1]
if OUT.exists() and any(OUT.iterdir()):
    raise SystemExit(f'{OUT} not empty')
OUT.mkdir(parents=True, exist_ok=True)
CACHE = HERE / 'cache'
CACHE.mkdir(exist_ok=True)

COL2LEVEL = {c: lv for lv, cols in CFG['name_columns'].items() if not lv.startswith('_') for c in cols}
NORM = CFG['normalize']
L1_RE = re.compile(NORM['L1_remove'])
L2_RES = [re.compile(p) for p in NORM['L2_strip_patterns']]


def head(s):
    return re.sub(r'\s', '', str(s or ''))


def l1(s):
    s = L1_RE.sub('', str(s))
    return s.lower() if NORM['L1_lower'] else s


def l2(s):
    s = str(s)
    while True:
        t = s
        for r in L2_RES:
            t = r.sub('', t)
        if t == s:
            return l1(s)
        s = t


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


def name_cols(header):
    return [(i, COL2LEVEL[head(h)], head(h)) for i, h in enumerate(header) if head(h) in COL2LEVEL]


YCFG = CFG['year']
YEAR_RE = re.compile(YCFG['name_pattern'])


def year_col(header):
    """index of the first configured year column present (config order = priority), else None"""
    hs = [head(h) for h in header]
    for c in YCFG['row_columns']:
        if c in hs:
            return hs.index(c), c
    return None, None


def year_of_text(s):
    ys = set(YEAR_RE.findall(str(s or '')))
    return int(ys.pop()) if len(ys) == 1 else None


def year_of_value(v):
    m = YEAR_RE.match(str(v).strip()) if v is not None else None
    return int(m.group(1)) if m else None


def fallback_year(unit, p):
    """(year, source) from the sheet/table name, then the file name; only when exactly one year appears"""
    y = year_of_text(unit)
    if y:
        return y, 'unit_name'
    y = year_of_text(p.name)
    return (y, 'file_name') if y else (None, 'none')


def from_xlsx(p):
    units = []
    wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
    for ws in wb.worksheets:
        rows = ws.iter_rows(values_only=True)
        cols, header = [], []
        for _ in range(CFG['header_scan_rows']):
            r = next(rows, None)
            if r is None:
                break
            cols, header = name_cols(r), r
            if cols:
                break
        if not cols:
            units.append({'unit': ws.title, 'status': '사업명 열 없음', 'names': {}})
            continue
        yi, yc = year_col(header)
        fy, fsrc = fallback_year(ws.title, p)
        cnt = Counter()
        for r in rows:
            ry = year_of_value(r[yi]) if yi is not None and yi < len(r) else None
            y, src = (ry, f'col:{yc}') if ry else (fy, fsrc)
            for i, lv, c in cols:
                v = r[i] if i < len(r) else None
                if v is not None and str(v).strip():
                    cnt[(lv, c, str(v).strip(), y, src)] += 1
        units.append({'unit': ws.title, 'status': 'ok', 'names': cnt})
    wb.close()
    return units


def from_csv(p):
    for enc in CFG['csv_encodings']:
        try:
            with open(p, encoding=enc, newline='') as f:
                rd = csv.reader(f)
                header = next(rd, [])
                cols = name_cols(header)
                if not cols:
                    return [{'unit': '', 'status': '사업명 열 없음', 'names': {}, 'encoding': enc}]
                yi, yc = year_col(header)
                fy, fsrc = fallback_year('', p)
                cnt = Counter()
                for r in rd:
                    ry = year_of_value(r[yi]) if yi is not None and yi < len(r) else None
                    y, src = (ry, f'col:{yc}') if ry else (fy, fsrc)
                    for i, lv, c in cols:
                        if i < len(r) and r[i].strip():
                            cnt[(lv, c, r[i].strip(), y, src)] += 1
                return [{'unit': '', 'status': 'ok', 'names': cnt, 'encoding': enc}]
        except UnicodeDecodeError:
            continue
    return [{'unit': '', 'status': '인코딩 실패', 'names': {}}]


def from_sqlite(p):
    units = []
    con = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
    kinds = ('table',) if CFG['sqlite_skip_views'] else ('table', 'view')
    q = f"select name from sqlite_master where type in ({','.join('?' * len(kinds))})"
    for (t,) in con.execute(q, kinds).fetchall():
        header = [r[1] for r in con.execute(f'pragma table_info("{t}")')]
        cols = [(None, COL2LEVEL[head(h)], h) for h in header if head(h) in COL2LEVEL]
        if not cols:
            units.append({'unit': t, 'status': '사업명 열 없음', 'names': {}})
            continue
        yi, yc = year_col(header)
        fy, fsrc = fallback_year(t, p)
        ysel = f'"{header[yi]}"' if yi is not None else 'NULL'
        cnt = Counter()
        for _, lv, c in cols:
            for v, yv, n in con.execute(f'select "{c}", {ysel}, count(*) from "{t}" group by 1, 2'):
                if v is not None and str(v).strip():
                    ry = year_of_value(yv)
                    y, src = (ry, f'col:{yc}') if ry else (fy, fsrc)
                    cnt[(lv, head(c), str(v).strip(), y, src)] += n
        units.append({'unit': t, 'status': 'ok', 'names': cnt})
    con.close()
    return units


def extract(p):
    st = p.stat()
    key = hashlib.sha256(f'v{CFG["extract_version"]}|{p}|{st.st_size}|{st.st_mtime_ns}'.encode()).hexdigest()[:24]
    cf = CACHE / f'{key}.json'
    if cf.exists():
        return json.loads(cf.read_text(encoding='utf-8'))
    ext = p.suffix.lower()
    try:
        units = from_xlsx(p) if ext in ('.xlsx', '.xlsm') else from_csv(p) if ext == '.csv' else from_sqlite(p)
        err = None
    except Exception as e:  # recorded, not hidden
        units, err = [], f'{type(e).__name__}: {e}'
    for u in units:
        u['names'] = [[lv, c, v, y, src, n] for (lv, c, v, y, src), n in u['names'].items()]
    rec = {'path': str(p), 'size': st.st_size, 'mtime_ns': st.st_mtime_ns, 'sha256': sha256(p), 'error': err, 'units': units}
    cf.write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
    return rec


root = Path(CFG['root'])
files = []
for top in sorted(root.iterdir()):
    if not top.is_dir() or top.name in CFG['exclude_top_dirs']:
        continue
    for dp, _, fs in os.walk(top):
        for f in sorted(fs):
            if Path(f).suffix.lower() in CFG['extensions']:
                files.append(Path(dp) / f)

file_rows, long_rows = [], []
for n, p in enumerate(files, 1):
    print(f'[{n}/{len(files)}] {p}', flush=True)
    rec = extract(p)
    is_copy = any(m in str(p) for m in CFG['copy_markers'])
    file_rows.append({k: rec[k] for k in ('path', 'size', 'mtime_ns', 'sha256', 'error')} | {
        'is_copy': is_copy, 'units': [{'unit': u['unit'], 'status': u['status'], 'distinct_names': len(u['names']),
                                       'rows': sum(x[5] for x in u['names'])} for u in rec['units']]})
    for u in rec['units']:
        for lv, c, v, y, src, cnt in u['names']:
            long_rows.append({'name': v, 'level': lv, 'column': c, 'year': y, 'year_source': src, 'count': cnt,
                              'file': rec['path'], 'unit': u['unit'], 'is_copy': is_copy})

# integrity: per-file row totals == long rows totals
assert sum(u['rows'] for f in file_rows for u in f['units']) == sum(r['count'] for r in long_rows)

TARGET = YCFG['target']
year_rows = Counter()
for r in long_rows:
    year_rows['판단불가' if r['year'] is None else str(r['year'])] += r['count']
keep = [r for r in long_rows if r['year'] == TARGET and len(l1(r['name'])) >= CFG['min_name_len']]
counts = {'files': len(files), 'files_error': sum(1 for f in file_rows if f['error']), 'files_copy': sum(f['is_copy'] for f in file_rows),
          'name_rows': sum(r['count'] for r in long_rows), 'name_rows_by_year': dict(sorted(year_rows.items())), 'target_year': TARGET,
          'kept_name_rows': sum(r['count'] for r in keep),
          'short_names_dropped': sum(r['count'] for r in long_rows if r['year'] == TARGET and len(l1(r['name'])) < CFG['min_name_len'])}
for scope, rows in (('전체', keep), ('사본 제외', [r for r in keep if not r['is_copy']])):
    c = {}
    for lv in ['전체 수준'] + sorted({r['level'] for r in rows}):
        sel = [r for r in rows if lv == '전체 수준' or r['level'] == lv]
        c[lv] = {'L0': len({r['name'] for r in sel}), 'L1': len({l1(r['name']) for r in sel}), 'L2': len({l2(r['name']) for r in sel})}
    counts[scope] = c

dedup = defaultdict(lambda: {'variants': Counter(), 'levels': Counter(), 'files': set()})
for r in keep:
    d = dedup[l2(r['name'])]
    d['variants'][r['name']] += r['count']
    d['levels'][r['level']] += r['count']
    d['files'].add(r['file'])
programs = [{'key_L2': k, 'n_variants': len(d['variants']), 'name': d['variants'].most_common(1)[0][0],
             'variants': dict(d['variants'].most_common()), 'levels': dict(d['levels']), 'n_files': len(d['files']), 'files': sorted(d['files'])}
            for k, d in sorted(dedup.items())]

# edge cases: L2 keys formed from >1 distinct L1 names (parenthetical stripped), largest first
l1_by_l2 = defaultdict(set)
for r in keep:
    l1_by_l2[l2(r['name'])].add(l1(r['name']))
merged = sorted(((k, sorted(v)) for k, v in l1_by_l2.items() if len(v) > 1), key=lambda x: -len(x[1]))
edge = {'L2_merged_groups': len(merged), 'samples': [{'key_L2': k, 'L1_names': v[:20], 'n': len(v)} for k, v in merged[:CFG['edge_samples']]]}

(OUT / 'files.json').write_text(json.dumps(file_rows, ensure_ascii=False, indent=1), encoding='utf-8')
with open(OUT / 'names_long.jsonl', 'w', encoding='utf-8') as f:
    for r in long_rows:
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
(OUT / 'programs_dedup.json').write_text(json.dumps(programs, ensure_ascii=False, indent=0), encoding='utf-8')
(OUT / 'edge_cases.json').write_text(json.dumps(edge, ensure_ascii=False, indent=1), encoding='utf-8')
counts['code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
counts['config_sha256'] = hashlib.sha256((HERE / 'config.json').read_bytes()).hexdigest()
counts['python'] = sys.version
(OUT / 'counts.json').write_text(json.dumps(counts, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps({k: v for k, v in counts.items() if k not in ('code_sha256', 'config_sha256', 'python')}, ensure_ascii=False, indent=1))
