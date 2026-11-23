"""runs/044 (2026-11-23): 공고명만 입력의 키워드(BM25) 실험용 Kiwi 토큰. 토크나이저는 tokenize_kiwi.py와 같은 함수
(harness_v2/harness.py kiwi_tokens, 'kiwi_space1_jabudam0_malware0', PUNCT 제외)를 그대로 import. 입력 = pool668 chunks.jsonl의 name.
출력 = name_enc_harness/kiwi_tokens_name_harness.json (키 = sha256(name)). 이미 있으면 중단.
usage: <KIWI_ENV_PYTHON> -X utf8 runs/044/name_enc_harness/tokenize_name_kiwi_harness.py"""
import hashlib, json, sys
from pathlib import Path
EXP = Path(__file__).resolve().parents[3]; O = EXP / 'runs/044/name_enc_harness/kiwi_tokens_name_harness.json'
assert not O.exists(), '이미 있음'
sys.path.insert(0, str(EXP)); import tokenize_kiwi as tk
C = [json.loads(l) for l in open(EXP / 'runs/044/pool668_newreview_harness/chunks.jsonl', encoding='utf-8')]; assert len(C) == 668
out = {}
for c in C:
    h = hashlib.sha256(c['name'].encode()).hexdigest()
    if h not in out: out[h] = tk.tok((h, c['name']))[1]
tmp = O.with_suffix('.tmp'); tmp.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8'); tmp.replace(O)
print('names', len(C), 'unique', len(out), 'empty', sum(not v for v in out.values()))
