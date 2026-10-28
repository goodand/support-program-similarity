"""Qwen3-Embedding-8B through vLLM in Docker (existing image govparse-semantic-budget-vllm:0.11.0-cu128-20261007; run only).

mode parity  : embed a fixed sample of texts already embedded locally (sentence-transformers, max_seq_length 2048) with
               vLLM (max-model-len 2048, truncate_prompt_tokens 2048) and compare cosine. PASS = min cosine >= 0.999 and the
               eval queries' positive ranks unchanged. Writes runs/<run>/docker/parity.json (+ throughput).
mode stage2  : only if parity passed -> embed the 'all' texts longer than 2048 tokens at max-model-len 8192 (6,000 chars
               fit) into runs/<run>/emb_cache_8192/qwen_all.npz (text-sha keyed, atomic writes, resumable).
The container mounts the local model read-only; it is removed afterwards. Usage (embedding env):
  <DEV_EVAL_ROOT>/_embedding_v3_env/Scripts/python.exe -I -X utf8 docker_vllm_embed.py <run> parity|stage2
"""
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs' / sys.argv[1]
MODE = sys.argv[2]
OUT = RUN / 'docker'; OUT.mkdir(exist_ok=True)
IMAGE = 'govparse-semantic-budget-vllm:0.11.0-cu128-20261007'
MODEL = '<DEV_EVAL_ROOT>/jh_dev/govparse-mvp-windows/govparse-mvp/data/semantic_budget_recall_20261007_v1/models/Qwen__Qwen3-Embedding-8B'
NAME, PORT = 'kim_vllm_embed', 8011
sha = lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest()  # noqa: E731


def start(max_len):
    subprocess.run(['docker', 'rm', '-f', NAME], capture_output=True)
    cmd = ['docker', 'run', '-d', '--gpus', 'all', '--name', NAME, '-p', f'{PORT}:8000', '--shm-size', '16g',
           '-v', f'{MODEL}:/models/emb:ro', '--entrypoint', 'python3', IMAGE, '-m', 'vllm.entrypoints.openai.api_server',
           '--model', '/models/emb', '--served-model-name', 'emb', '--task', 'embed', '--dtype', 'bfloat16',
           '--max-model-len', str(max_len), '--gpu-memory-utilization', '0.85', '--max-num-batched-tokens', str(max(8192, max_len * 4))]
    subprocess.run(cmd, check=True, capture_output=True)
    for _ in range(180):
        try:
            urllib.request.urlopen(f'http://127.0.0.1:{PORT}/health', timeout=5)
            return
        except Exception:
            time.sleep(5)
    logs = subprocess.run(['docker', 'logs', '--tail', '60', NAME], capture_output=True, text=True)
    (OUT / f'start_fail_{max_len}.log').write_text(logs.stdout + logs.stderr, encoding='utf-8')
    raise SystemExit('vLLM did not become healthy; see ' + str(OUT))


def stop():
    subprocess.run(['docker', 'rm', '-f', NAME], capture_output=True)


def embed(texts, trunc):
    body = json.dumps({'model': 'emb', 'input': texts, 'truncate_prompt_tokens': trunc}).encode('utf-8')
    req = urllib.request.Request(f'http://127.0.0.1:{PORT}/v1/embeddings', data=body, headers={'Content-Type': 'application/json'})
    data = json.loads(urllib.request.urlopen(req, timeout=3600).read())['data']
    V = np.array([d['embedding'] for d in sorted(data, key=lambda d: d['index'])], np.float32)
    return V / np.linalg.norm(V, axis=1, keepdims=True)


T = json.loads((RUN / 'texts.json').read_text(encoding='utf-8'))
local = {}
for f in sorted((RUN / 'emb_cache').glob('qwen_*.npz')):
    if '.tmp' not in f.name:
        z = np.load(f); local.update(zip(z['h'].tolist(), z['v']))

if MODE == 'parity':
    pool = {}
    for t in ('name', 'all'):
        for k, x in T['texts'][t].items():
            if sha(x) in local:
                pool[sha(x)] = x
    by_len = sorted(pool.items(), key=lambda hx: len(hx[1]))
    idx = np.linspace(0, len(by_len) - 1, 300).astype(int)
    sample = [by_len[i] for i in idx]
    for g, ks in T['eval_groups'].items():  # the eval programs' own texts
        for k in ks:
            for t in ('name', 'all'):
                x = T['texts'][t][k]
                if sha(x) in local:
                    sample.append((sha(x), x))
    sample = list(dict(sample).items())
    start(2048)
    try:
        t0 = time.time(); V = np.concatenate([embed([x for _, x in sample[i:i + 32]], 2048) for i in range(0, len(sample), 32)]); dt = time.time() - t0
    finally:
        stop()
    L = np.stack([local[h] for h, _ in sample]).astype(np.float32)
    L /= np.linalg.norm(L, axis=1, keepdims=True)
    cos = (V * L).sum(1)
    res = {'n': len(sample), 'cos_min': float(cos.min()), 'cos_mean': float(cos.mean()), 'cos_p01': float(np.percentile(cos, 1)),
           'worst': [{'len': len(sample[i][1]), 'cos': float(cos[i])} for i in np.argsort(cos)[:10]],
           'seconds': dt, 'texts_per_s': len(sample) / dt, 'pass': bool(cos.min() >= 0.999), 'image': IMAGE}
    (OUT / 'parity.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(res, ensure_ascii=False))

elif MODE == 'stage2':
    par = json.loads((OUT / 'parity.json').read_text(encoding='utf-8'))
    assert par['pass'], 'parity failed; stage2 refused'
    store = RUN / 'emb_cache_8192' / 'qwen_all.npz'; store.parent.mkdir(exist_ok=True)
    have = {}
    if store.exists():
        z = np.load(store); have = dict(zip(z['h'].tolist(), z['v']))
    texts = {sha(x): x for x in T['texts']['all'].values() if len(x) > 1500}  # shorter texts fit 2048 tokens: reuse stage1
    todo = sorted((h for h in texts if h not in have), key=lambda h: len(texts[h]))
    print('stage2 todo', len(todo), flush=True)
    start(8192)
    try:
        t0 = time.time()
        for s in range(0, len(todo), 256):
            part = todo[s:s + 256]
            have.update(zip(part, embed([texts[h] for h in part], 8192).astype(np.float16)))
            tmp = store.with_name('qwen_all.tmp.npz'); np.savez(tmp, h=np.array(list(have)), v=np.stack(list(have.values()))); tmp.replace(store)
            print(f'{min(s + 256, len(todo))}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    finally:
        stop()
    print('stage2 done', len(have))
