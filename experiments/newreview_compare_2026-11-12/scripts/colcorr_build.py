"""열 상관 분석 입력 만들기: 668 공고 x (열 10개 값만, 9칸에서 열 하나씩 뺀 9조합) 텍스트를 한 jsonl로.
9칸 재구성이 기존 chunks.jsonl text와 668개 모두 같아야 진행(같은 형식 보장).
usage: .venv/Scripts/python.exe -X utf8 runs/044/colcorr_build_harness.py
출력: runs/044/colcorr_harness/{texts.jsonl, req.json, index.json}"""
import os
import json, hashlib, openpyxl
from pathlib import Path
O = Path('runs/044/colcorr_harness'); O.mkdir(exist_ok=True)
wb = openpyxl.load_workbook(os.environ['NOTICE_MASTER_XLSX'], read_only=True)
ws = [wb[s] for s in wb.sheetnames if s.startswith(os.environ['NOTICE_MASTER_SHEET_PREFIX'])][0]
rows = list(ws.iter_rows(values_only=True)); H = rows[1]
R = [json.loads(l) for l in open('runs/044/pool668_newreview_harness/chunks.jsonl', encoding='utf-8')]
NINE = ['26공고명', '기관', '대분류', '중분류', '업종', '목적', '내용', '대상', '대상유형']
COL = {c: [i for i in range(128, 152) if H[i] == c][0] for c in NINE + ['지원규모']}
val = lambda row, c: ('' if row[COL[c]] in (None, '-') else str(row[COL[c]]).strip())
def nine(row, drop=None):
    return ' / '.join(f'{c}: {val(row, c)}' for c in NINE if c != drop and val(row, c))
srow = [rows[r['id']['sheet_row'] - 1] for r in R]
ok = sum(nine(s) == r['text'] for s, r in zip(srow, R))
print('9칸 재구성 일치', ok, '/ 668')
assert ok == 668, '재구성 형식이 기존 청크와 다름 -> 중단'
sets = {f'col:{c}': [val(s, c) for s in srow] for c in NINE + ['지원규모']}
sets.update({f'drop:{c}': [nine(s, c) for s in srow] for c in NINE})
tx, idx = {}, {}
for k, L in sets.items():
    hs = []
    for t in L:
        h = hashlib.sha256(t.encode()).hexdigest() if t else None
        if t: tx.setdefault(h, t)
        hs.append(h)
    idx[k] = hs
open(O / 'texts.jsonl', 'w', encoding='utf-8').write('\n'.join(json.dumps({'h': h, 't': t}, ensure_ascii=False) for h, t in tx.items()) + '\n')
json.dump(idx, open(O / 'index.json', 'w'))
s = hashlib.sha256((O / 'texts.jsonl').read_bytes()).hexdigest()
json.dump({'texts_file_sha256': s, 'encode_max_seq_length': 2048, 'n_texts': len(tx)}, open(O / 'req.json', 'w'))
print('고유 텍스트', len(tx), 'req_sha', hashlib.sha256((O / 'req.json').read_bytes()).hexdigest())
print({k: sum(h is None for h in v) for k, v in idx.items() if any(h is None for h in v)})
