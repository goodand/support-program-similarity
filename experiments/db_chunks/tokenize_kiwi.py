"""Kiwi tokens for every unique chunk text in runs/<run>/texts.json, with the analyzer of the earlier best BM25 runs.

Tokenizer is not re-implemented: harness_v2/harness.py kiwi_tokens (analyzer 'kiwi_space1_jabudam0_malware0', PUNCT
tags dropped) is imported and called as-is. Parallel over texts (multiprocessing). Output runs/<run>/kiwi_tokens.json
keyed by text sha256. Usage (program_mapping_nlp_v1 venv, which has kiwipiepy 0.24.0):
  <DEV_EVAL_ROOT>/jh_dev/program_mapping_nlp_v1/.venv/Scripts/python.exe -X utf8 tokenize_kiwi.py <run> [workers]
"""
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path

HV2 = Path('<DEV_EVAL_ROOT>/jh_dev/program_mapping_nlp_v1/harness_v2')
HERE = Path(__file__).resolve().parent
_hz = None


def tok(item):
    global _hz
    if _hz is None:
        sys.path.insert(0, str(HV2))
        import harness as hz
        _hz = hz
    h, text = item
    return h, _hz.kiwi_tokens(text) if text.strip() else []


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    RUN = HERE / 'runs' / sys.argv[1]
    W = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    T = json.loads((RUN / 'texts.json').read_text(encoding='utf-8'))
    uniq = {}
    for per in T['texts'].values():
        for x in per.values():
            uniq.setdefault(hashlib.sha256(x.encode('utf-8')).hexdigest(), x)
    out_p = RUN / 'kiwi_tokens.json'
    done = json.loads(out_p.read_text(encoding='utf-8')) if out_p.exists() else {}
    todo = sorted(((h, x) for h, x in uniq.items() if h not in done), key=lambda hx: -len(hx[1]))
    print('unique texts', len(uniq), 'todo', len(todo), flush=True)
    with Pool(W) as pool:
        for n, (h, ts) in enumerate(pool.imap_unordered(tok, todo, chunksize=16), 1):
            done[h] = ts
            if n % 20000 == 0:
                tmp = out_p.with_suffix('.tmp'); tmp.write_text(json.dumps(done, ensure_ascii=False), encoding='utf-8'); tmp.replace(out_p)
                print(n, flush=True)
    tmp = out_p.with_suffix('.tmp'); tmp.write_text(json.dumps(done, ensure_ascii=False), encoding='utf-8'); tmp.replace(out_p)
    print('done', len(done), hashlib.sha256(out_p.read_bytes()).hexdigest())
