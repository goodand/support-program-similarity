"""Qwen3-Embedding-8B: jsonl {"h": sha256, "t": text} -> <out>/s#####.npz shards (h: hex array, v: float16 N x 4096, L2-normalized).

Matches the Windows reference (sentence-transformers, bf16, sdpa, left padding, max_seq_length 2048, last-token pooling,
normalize, no document instruction). 4,096 texts per shard, tmp write then rename, hashes already in <out> are skipped.

backends
  st    sentence-transformers exactly as the reference (batch 64 if len < 1500 chars else 16, texts sorted by length).
  kv    shared-prefix KV reuse: texts sorted lexicographically, greedy groups whose token-level common prefix is
        >= --min-prefix tokens; the prefix is run once, then only each text's suffix is run on a copy of its KV cache.
        Causal attention + last-token pooling make this exact up to float error.
  vllm  vLLM offline LLM.embed (run inside the vllm image), optional --prefix-caching.

Usage: python embed_shards.py <in.jsonl> <out_dir> [--backend st|kv|vllm] [--hashes FILE] [--report FILE] ...
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

MODEL = '/models/Qwen__Qwen3-Embedding-8B'
SHARD = 4096
log = lambda *a: print(time.strftime('%m-%d %H:%M:%S'), *a, flush=True)  # noqa: E731


def have_hashes(out):
    hs = set()
    for p in sorted(out.glob('s*.npz')):
        if '.tmp' in p.name:
            continue
        with np.load(p) as z:
            hs.update(z['h'].tolist())
    return hs


def write_shard(out, idx, hs, v):
    p = out / f's{idx:05d}.npz'
    tmp = p.with_name(p.stem + '.tmp.npz')
    np.savez(tmp, h=np.array(hs), v=v.astype(np.float16))
    tmp.replace(p)


def load_st():
    import torch
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL, device='cuda', local_files_only=True, model_kwargs={'torch_dtype': torch.bfloat16, 'attn_implementation': 'sdpa'},
                            tokenizer_kwargs={'padding_side': 'left'})
    m.max_seq_length = 2048
    return m


def embed_st(m, texts):
    L = len(texts[-1]) if texts else 0
    return m.encode(texts, batch_size=64 if L < 1500 else 16, normalize_embeddings=True, convert_to_numpy=True)


def kv_groups(ids, min_prefix):
    """Greedy groups over lexicographically sorted token lists; returns [(member indices, prefix length)]."""
    order = sorted(range(len(ids)), key=lambda i: ids[i])
    groups, i = [], 0
    while i < len(order):
        a = ids[order[i]]; p = len(a) - 1; j = i + 1
        while j < len(order):
            b = ids[order[j]]; n = min(p, len(b) - 1); k = 0
            while k < n and a[k] == b[k]:
                k += 1
            if k < min_prefix:
                break
            p = k; j += 1
        groups.append((order[i:j], p if j - i > 1 else 0))
        i = j
    return groups


def embed_kv(m, texts, min_prefix, max_batch_tokens, stats):
    import copy
    import torch
    tok = m.tokenizer
    model = m[0].auto_model
    ids = [tok(t, truncation=True, max_length=m.max_seq_length)['input_ids'] for t in texts]
    out = np.zeros((len(texts), model.config.hidden_size), np.float32)
    dev = model.device

    def run(batch, past, plen):
        S = max(len(s) for s in batch)
        x = torch.full((len(batch), S), tok.pad_token_id, dtype=torch.long)
        am = torch.zeros((len(batch), plen + S), dtype=torch.long)
        am[:, :plen] = 1
        for r, s in enumerate(batch):
            x[r, :len(s)] = torch.tensor(s); am[r, plen:plen + len(s)] = 1  # right padding: pads come after the pooled token
        pos = torch.arange(plen, plen + S).expand(len(batch), S)
        h = model(input_ids=x.to(dev), attention_mask=am.to(dev), position_ids=pos.to(dev), past_key_values=past, use_cache=past is not None).last_hidden_state
        last = torch.tensor([len(s) - 1 for s in batch], device=dev)
        return h[torch.arange(len(batch), device=dev), last].float()

    with torch.inference_mode():
        for mem, plen in kv_groups(ids, min_prefix):
            stats['tokens_full'] += sum(len(ids[i]) for i in mem)
            if plen == 0:  # no shared prefix: plain batched forward, one text per call group
                for s in range(0, len(mem), 16):
                    part = mem[s:s + 16]
                    out[part] = run([ids[i] for i in part], None, 0).cpu().numpy()
                stats['tokens_run'] += sum(len(ids[i]) for i in mem)
                continue
            pre = model(input_ids=torch.tensor([ids[mem[0]][:plen]], device=dev), use_cache=True).past_key_values
            stats['tokens_run'] += plen
            sufs = [ids[i][plen:] for i in mem]
            s = 0
            while s < len(mem):
                e = s + 1
                while e < len(mem) and (e - s + 1) * (plen + max(len(x) for x in sufs[s:e + 1])) <= max_batch_tokens:
                    e += 1
                past = copy.deepcopy(pre)
                past.batch_repeat_interleave(e - s)
                out[mem[s:e]] = run(sufs[s:e], past, plen).cpu().numpy()
                stats['tokens_run'] += sum(len(x) for x in sufs[s:e])
                s = e
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('inp'); ap.add_argument('out')
    ap.add_argument('--backend', choices=('st', 'kv', 'vllm'), default='st')
    ap.add_argument('--hashes', help='file with one hash per line: embed only these')
    ap.add_argument('--report', help='write a JSON speed/stat report here')
    ap.add_argument('--min-prefix', type=int, default=16, help='kv: min shared prefix tokens to form a group')
    ap.add_argument('--max-batch-tokens', type=int, default=65536, help='kv: batch size limit in (prefix+suffix) tokens')
    ap.add_argument('--prefix-caching', action='store_true', help='vllm: enable_prefix_caching')
    ap.add_argument('--gpu-mem', type=float, default=0.85, help='GPU memory cap as a fraction (vllm: gpu_memory_utilization; st/kv: per-process cap)')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    keep = set(Path(a.hashes).read_text(encoding='utf-8').split()) if a.hashes else None
    by = {}
    for line in open(a.inp, encoding='utf-8'):
        o = json.loads(line)
        if keep is None or o['h'] in keep:
            by[o['h']] = o['t']
    done = have_hashes(out)
    todo = sorted((h for h in by if h not in done), key=lambda h: len(by[h]))
    chars = sum(len(by[h]) for h in todo)
    log('embed: input', len(by), 'cached', len(done), 'todo', len(todo), 'chars', chars, 'backend', a.backend)
    if not todo:
        return
    n0 = len([p for p in out.glob('s*.npz') if '.tmp' not in p.name])
    stats = {'tokens_full': 0, 'tokens_run': 0}
    if a.backend == 'vllm':
        from vllm import LLM
        llm = LLM(model=MODEL, task='embed', dtype='bfloat16', max_model_len=2048, gpu_memory_utilization=a.gpu_mem,
                  enable_prefix_caching=a.prefix_caching, disable_log_stats=False)
        enc = lambda T: np.array([o.outputs.embedding for o in llm.embed(T, truncate_prompt_tokens=2048, use_tqdm=False)], np.float32)  # noqa: E731
    else:
        import torch
        torch.cuda.set_per_process_memory_fraction(a.gpu_mem)  # share the GPU with other jobs: hard cap for this process
        m = load_st()
        enc = (lambda T: embed_st(m, T)) if a.backend == 'st' else (lambda T: embed_kv(m, T, a.min_prefix, a.max_batch_tokens, stats))  # noqa: E731
    if a.backend == 'kv':  # group across the whole input so prefixes are shared beyond shard borders
        todo = sorted(todo, key=lambda h: by[h])
    t0 = time.time(); c0 = 0
    for s in range(0, len(todo), SHARD):
        part = todo[s:s + SHARD]
        v = enc([by[h] for h in part])
        v = v / np.linalg.norm(v, axis=1, keepdims=True)
        write_shard(out, n0 + s // SHARD, part, v)
        c0 += sum(len(by[h]) for h in part); el = time.time() - t0
        log(f'embed {min(s + SHARD, len(todo))}/{len(todo)} {c0 / el:,.0f} chars/s')
    el = time.time() - t0
    rep = {'backend': a.backend, 'n': len(todo), 'chars': chars, 'seconds': el, 'chars_per_s': chars / el, **stats,
           'prefix_caching': a.prefix_caching if a.backend == 'vllm' else None, 'min_prefix': a.min_prefix if a.backend == 'kv' else None}
    if a.backend == 'vllm':  # prefix cache counters (V1 engine metrics), if this vLLM version exposes them
        try:
            rep['vllm_prefix_cache'] = {mt.name: getattr(mt, 'value', None) for mt in llm.get_metrics() if 'prefix_cache' in mt.name}
        except Exception as ex:  # noqa: BLE001
            rep['vllm_prefix_cache'] = f'unavailable: {type(ex).__name__}: {ex}'
    if stats['tokens_full']:
        rep['token_saving'] = 1 - stats['tokens_run'] / stats['tokens_full']
    log('report', json.dumps(rep))
    if a.report:
        Path(a.report).write_text(json.dumps(rep, indent=1), encoding='utf-8')


if __name__ == '__main__':
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    main()
