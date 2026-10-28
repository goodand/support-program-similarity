"""Deterministic full-text scan for eval_name_match_001 (fixes from the adversarial harness review, 2026-10-28).

For each of the 54 eval programs, find every place in provider A's central data where the program name occurs, in ANY text:
tabular cells (xlsx data_only, csv, sqlite tables+views, text and blob), hwpx section XML, PDF text, json strings,
txt, and the same inside zip archives. Normalization = l1/l2 imported from merge_program_names.py (same as the merge).
Keys per program: l2 (brackets removed), l1 of the full name (brackets kept as text), l1 of each bracket's inner text
(>= min_key_len). Direction: key in cell/doc text; for name columns also the reverse (cell l2 in key, len >= min_key_len).
Hits are aggregated per (program, file, unit, column, year) with a count and one sample — no cap. Per-file results are
cached (cache/<key>.json) so the run resumes. Errors are recorded per file/unit. Usage:
  .venv/Scripts/python.exe -I -X utf8 harness/eval_name_match_001/fullscan.py <out_dir_name>
"""
import ast
import csv
import hashlib
import io
import json
import os
import re
import sqlite3
import sys
import zipfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT_PROJ = HERE.parents[1]
SRC = ROOT_PROJ / 'merge_program_names.py'
CFG = json.loads((ROOT_PROJ / 'config.json').read_text(encoding='utf-8'))
SCAN = {
    'root': '<DEV_EVAL_ROOT>',
    'exclude_top_dirs': CFG['exclude_top_dirs'],
    '_doc_local': 'local-government data is out of scope (eval programs are central). Listed explicitly; excluded files are written to coverage.json.',
    'exclude_dirs': ['<DEV_EVAL_ROOT>/<지자체_DB_폴더>'],
    'exclude_files': ['<DEV_EVAL_ROOT>/<중기_DB_폴더>/<사업정보_DB>.db',
                      '<DEV_EVAL_ROOT>/<중기_DB_폴더>/<사업정보_DB_이전본>.db'],
    'exts': ['.xlsx', '.xlsm', '.csv', '.sqlite', '.db'],
    '_doc_exts': '사용자 결정(2026-10-28): 자료 제공자 A DB·CSV·엑셀만 본다. hwpx·hwp·pdf·json·zip은 검색하지 않는다(not_scanned_by_ext에 개수 기록).',
    'zip_inner_exts': ['.hwpx', '.hwp', '.pdf', '.xml', '.txt', '.csv', '.json'],
    'min_key_len': 6,
}
EVAL = Path('<DEV_EVAL_ROOT>/jh_dev/<EXPERIMENT_DIR>/runs/002/eval_match.json')
EVAL_SHA = '58f0259015edaeed4b5fc1cf4fd9e7fe063ced5a9d3da218579fe94f1a842d56'

ns = {'re': re, 'CFG': CFG, 'NORM': CFG['normalize'], 'YCFG': CFG['year'],
      'L1_RE': re.compile(CFG['normalize']['L1_remove']), 'L2_RES': [re.compile(p) for p in CFG['normalize']['L2_strip_patterns']],
      'YEAR_RE': re.compile(CFG['year']['name_pattern'])}
NEED = {'l1', 'l2', 'head', 'year_col', 'year_of_text', 'year_of_value'}
exec(compile(ast.Module(body=[n for n in ast.parse(SRC.read_text(encoding='utf-8')).body
                              if isinstance(n, ast.FunctionDef) and n.name in NEED], type_ignores=[]), str(SRC), 'exec'), ns)
l1, l2, head, year_col, year_of_text, year_of_value = (ns[k] for k in ('l1', 'l2', 'head', 'year_col', 'year_of_text', 'year_of_value'))
COL2LEVEL = {c: lv for lv, cols in CFG['name_columns'].items() if not lv.startswith('_') for c in cols}
MINK = SCAN['min_key_len']


def build_keys():
    assert hashlib.sha256(EVAL.read_bytes()).hexdigest() == EVAL_SHA, 'eval_match.json changed'
    progs = json.loads(EVAL.read_text(encoding='utf-8'))['rows']
    keys = []
    for i, o in enumerate(progs):
        ks = {('l2', l2(o['사업명']))}
        ks.add(('full_l1', l1(o['사업명'])))
        for inner in re.findall(r'[\(\[（【]([^\)\]）】]*)[\)\]）】]', o['사업명']):
            if len(l1(inner)) >= MINK:
                ks.add(('bracket_inner', l1(inner)))
        keys.append({'i': i, 'name': o['사업명'], 'group': o['그룹ID'], 'keys': sorted(ks)})
    return keys


KEYS = build_keys()


class Acc:
    def __init__(self):
        self.h = defaultdict(lambda: [0, ''])

    def text(self, v, unit, col, year, is_name_col):
        if v is None:
            return
        if isinstance(v, bytes):
            v = v.decode('utf-8', 'replace')
        v = str(v)
        s = l1(v)
        if len(s) < CFG['min_name_len']:
            return
        s2 = l2(v) if is_name_col else None
        for k in KEYS:
            for kind, key in k['keys']:
                hit = key and key in s
                rev = bool(is_name_col and len(s2) >= MINK and s2 in key and s2 != key)
                if hit or rev:
                    e = self.h[(k['i'], unit, col, year, kind if hit else 'reverse_' + kind, bool(is_name_col))]
                    e[0] += 1
                    if not e[1]:
                        j = max(0, s.find(key) if hit else 0)
                        e[1] = v[:160] if len(v) <= 160 else '…' + re.sub(r'\s+', ' ', v)[max(0, j - 40): j + 120]

    def rows(self):
        return [{'prog': i, 'unit': u, 'column': c, 'year': y, 'match': kind, 'is_name_col': nc, 'count': n, 'sample': smp}
                for (i, u, c, y, kind, nc), (n, smp) in self.h.items()]


def scan_sqlite(p, acc, errs):
    con = sqlite3.connect(f'file:{p}?mode=ro', uri=True)
    for t, typ in con.execute("select name, type from sqlite_master where type in ('table','view')").fetchall():
        try:
            header = [r[1] for r in con.execute(f'pragma table_info("{t}")')]
            yi, yc = year_col(header)
            ysel = f'"{header[yi]}"' if yi is not None else 'NULL'
            fy = year_of_text(t) or year_of_text(Path(p).name)
            for c in header:
                nc = head(c) in COL2LEVEL
                for v, yv in con.execute(f'select distinct "{c}", {ysel} from "{t}" where typeof("{c}") in (\'text\',\'blob\')'):
                    acc.text(v, f'{typ}:{t}', c, year_of_value(yv) or fy, nc)
        except Exception as e:
            errs.append({'unit': t, 'error': f'{type(e).__name__}: {e}'})
    con.close()


def scan_rows(rows, unit, fname, acc):
    header, yi = None, None
    for r in rows:
        if header is None:
            header = [str(x) if x is not None else '' for x in r]
            yi, _ = year_col(header)
            fy = year_of_text(unit) or year_of_text(fname)
            ncs = [head(h) in COL2LEVEL for h in header]
        ry = (year_of_value(r[yi]) if yi is not None and yi < len(r) else None) or fy
        for j, v in enumerate(r):
            if isinstance(v, (str, bytes)):
                acc.text(v, unit, header[j] if j < len(header) else f'col{j}', ry, ncs[j] if j < len(ncs) else False)


def hwp5_text(data):
    """body text of an HWP5 (OLE) document: BodyText/Section* records with tag 67 (PARA_TEXT), UTF-16LE"""
    import olefile
    import struct
    import zlib
    ole = olefile.OleFileIO(io.BytesIO(data))
    compressed = bool(struct.unpack('<I', ole.openstream('FileHeader').read()[36:40])[0] & 1)
    out = []
    for s in sorted(('/'.join(e) for e in ole.listdir() if e[0] == 'BodyText'), key=lambda x: int(re.sub(r'\D', '', x) or 0)):
        b = ole.openstream(s).read()
        if compressed:
            b = zlib.decompress(b, -15)
        i = 0
        while i + 4 <= len(b):
            h = struct.unpack('<I', b[i:i + 4])[0]
            tag, size = h & 0x3FF, (h >> 20) & 0xFFF
            i += 4
            if size == 0xFFF:
                size = struct.unpack('<I', b[i:i + 4])[0]
                i += 4
            if tag == 67:
                out.append(b[i:i + size].decode('utf-16-le', 'replace'))
            i += size
    return re.sub(r'[\x00-\x1f]', ' ', ' '.join(out))


def scan_bytes(name, data, acc, errs, unit_prefix=''):
    """document-type content from bytes (used for files and zip members)"""
    ext = Path(name).suffix.lower()
    y = year_of_text(name)
    if ext in ('.hwpx', '.hwp') and data[:4] == b'\xd0\xcf\x11\xe0':  # HWP5 binary, whatever the extension says
        acc.text(hwp5_text(data), unit_prefix + 'hwp5', 'text', y, False)
    elif ext == '.hwpx':
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            for m in sorted(z.namelist()):
                if m.startswith('Contents/section') and m.endswith('.xml'):
                    txt = re.sub(r'<[^>]+>', ' ', z.read(m).decode('utf-8', 'replace'))
                    acc.text(txt, unit_prefix + m, 'text', y, False)
    elif ext == '.pdf':
        import pymupdf
        d = pymupdf.open(stream=data, filetype='pdf')
        acc.text(' '.join(pg.get_text() for pg in d), unit_prefix + 'pdf', 'text', y, False)
    elif ext == '.json':
        def walk(o):
            if isinstance(o, dict):
                for v in o.values():
                    yield from walk(v)
            elif isinstance(o, list):
                for v in o:
                    yield from walk(v)
            elif isinstance(o, str):
                yield o
        for s in walk(json.loads(data.decode('utf-8-sig', 'replace'))):
            acc.text(s, unit_prefix + 'json', 'string', y, False)
    elif ext in ('.txt', '.xml'):
        acc.text(data.decode('utf-8', 'replace') if b'\x00' not in data[:1000] else '', unit_prefix + ext, 'text', y, False)
    elif ext == '.csv':
        txt = None
        for enc in CFG['csv_encodings']:
            try:
                txt = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if txt is None:
            errs.append({'unit': unit_prefix, 'error': 'csv 인코딩 실패'})
        else:
            scan_rows(csv.reader(io.StringIO(txt)), unit_prefix + 'csv', name, acc)


def scan_file(p):
    st = os.stat(p)
    key = hashlib.sha256(f'fs1|{EVAL_SHA}|{json.dumps(SCAN, sort_keys=True)}|{p}|{st.st_size}|{st.st_mtime_ns}'.encode()).hexdigest()[:24]
    cf = HERE / 'cache' / f'{key}.json'
    if cf.exists():
        return json.loads(cf.read_text(encoding='utf-8'))
    acc, errs = Acc(), []
    ext = Path(p).suffix.lower()
    try:
        if ext in ('.sqlite', '.db'):
            scan_sqlite(p, acc, errs)
        elif ext in ('.xlsx', '.xlsm'):
            import openpyxl
            wb = openpyxl.load_workbook(p, read_only=True, data_only=True)
            for ws in wb.worksheets:
                scan_rows(ws.iter_rows(values_only=True), ws.title, Path(p).name, acc)
            wb.close()
        elif ext == '.zip':
            with zipfile.ZipFile(p) as z:
                for m in z.namelist():
                    if Path(m).suffix.lower() in SCAN['zip_inner_exts']:
                        try:
                            scan_bytes(m, z.read(m), acc, errs, unit_prefix=f'zip:{m}:')
                        except Exception as e:
                            errs.append({'unit': m, 'error': f'{type(e).__name__}: {e}'})
        else:
            scan_bytes(Path(p).name, Path(p).read_bytes(), acc, errs)
    except Exception as e:
        errs.append({'unit': '', 'error': f'{type(e).__name__}: {e}'})
    rec = {'path': p, 'ext': ext, 'size': st.st_size, 'errors': errs, 'hits': acc.rows()}
    tmp = cf.with_suffix('.tmp')
    tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding='utf-8')
    tmp.replace(cf)
    return rec


def list_files():
    files, excluded, skipped = [], [], defaultdict(int)
    ex_dirs = [Path(d) for d in SCAN['exclude_dirs']]
    ex_files = {str(Path(f)) for f in SCAN['exclude_files']}
    root = Path(SCAN['root'])
    for top in sorted(root.iterdir()):
        if not top.is_dir() or top.name in SCAN['exclude_top_dirs']:
            continue
        for dp, _, fs in os.walk(top):
            for f in sorted(fs):
                p = Path(dp) / f
                if any(d in p.parents for d in ex_dirs) or str(p) in ex_files:
                    excluded.append(str(p))
                elif f.startswith('~$'):
                    skipped['~$ 임시파일'] += 1
                elif p.suffix.lower() in SCAN['exts']:
                    files.append(str(p))
                else:
                    skipped[p.suffix.lower() or '(없음)'] += 1
    return files, excluded, dict(skipped)


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    OUT = HERE / sys.argv[1]
    if OUT.exists() and any(OUT.iterdir()):
        raise SystemExit(f'{OUT} not empty')
    OUT.mkdir(parents=True, exist_ok=True)
    (HERE / 'cache').mkdir(exist_ok=True)
    files, excluded, skipped = list_files()
    print('files', len(files), 'excluded', len(excluded), 'skipped', skipped, flush=True)
    big = [f for f in files if os.path.getsize(f) > 200_000_000]
    small = [f for f in files if f not in set(big)]
    recs = []
    with Pool(16) as pool:
        for n, r in enumerate(pool.imap_unordered(scan_file, small + big, chunksize=8), 1):
            recs.append(r)
            if n % 2000 == 0 or n == len(files):
                print(f'{n}/{len(files)}', flush=True)
    cov = defaultdict(lambda: {'files': 0, 'files_with_error': 0})
    for r in recs:
        cov[r['ext']]['files'] += 1
        cov[r['ext']]['files_with_error'] += bool(r['errors'])
    hits = []
    for r in recs:
        for h in r['hits']:
            hits.append({**h, 'program': KEYS[h['prog']]['name'], 'group': KEYS[h['prog']]['group'], 'file': r['path']})
    errors = [{'file': r['path'], **e} for r in recs for e in r['errors']]
    coverage = {'scanned_by_ext': dict(cov), 'not_scanned_by_ext': skipped, 'excluded_local': excluded, 'n_errors': len(errors),
                'scan_config': SCAN, 'keys': KEYS, 'eval_sha256': EVAL_SHA,
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'merge_code_sha256': hashlib.sha256(SRC.read_bytes()).hexdigest()}
    for name, obj in (('hits.json', hits), ('errors.json', errors), ('coverage.json', coverage)):
        tmp = OUT / (name + '.tmp')
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
        tmp.replace(OUT / name)
    found = {h['prog'] for h in hits if not h['match'].startswith('reverse')}
    print('programs with a forward hit:', len(found), '/', len(KEYS), '| errors', len(errors))
    print(json.dumps(dict(cov), ensure_ascii=False))
