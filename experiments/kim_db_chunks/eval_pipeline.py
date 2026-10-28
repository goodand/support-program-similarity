"""Retrieval pipeline grid on the 2026 central pool (RAG retrieval stage), reusing the earlier best components.

Chunk conditions (texts.json): N_name, A_all, B_split (목적·내용 / 수혜대상), C_split_legal (+근거법령),
  D_nam_all / D_nam_content / D_nam_content_nomerge (eval-group programs carry 남윤형 엑셀 text; others keep Kim A_all).
Stage 1 (full order):
  qwen        Qwen3-Embedding-8B cosine (same model/loading as before; embeddings cached by text sha256)
  vn          BM25 with two-stage verbosity/scope normalization (Na 2015) = exp_M05_VN.vn_field, bv=1, bs=0.75, k1=1.5,
              on Kiwi tokens (harness_v2 kiwi_tokens). Vectorized; equality with the exec'd original is asserted on a sample.
  vn_sym      0.5*(vn(q->c) + vn(c->q))  (BM25 양방향 평균)
  qwen+vn, qwen+vn_sym   RRF k=60 of the two orders (harness.rrf)
Stage 2 on the stage-1 top 30 (exp_recombine.stage2 rules, tail kept):
  none | rrf60_reranker (1/(60+base)+1/(60+reranker)) | reciprocal (0.8/(60+base)+0.2/(60+reverse_rank), reverse rank of the
  query among the candidate's Qwen neighbours over pool+query) | rrf60_reranker->reciprocal
Reranker = Qwen3-Reranker-0.6B, protocol/instruction/prefix/suffix of score_reranker_full.py (verbatim core), cached per pair.
Multi-chunk conditions: per-type scores, mean over types present in both programs (fallback: name-only score); vn_sym and
reverse rank use the same combination. Pre-declared primary (earlier best): qwen+vn|rrf60_reranker.
Rank = tie-averaged rank of each positive with the query removed. Usage (embedding env):
  <DEV_EVAL_ROOT>/_embedding_v3_env/Scripts/python.exe -I -X utf8 eval_pipeline.py <run> <out_name>
"""
import ast
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

os.environ['HF_HUB_OFFLINE'] = '1'; os.environ['TRANSFORMERS_OFFLINE'] = '1'
sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs' / sys.argv[1]
OUT = RUN / sys.argv[2]
OUT.mkdir(exist_ok=True)
HV2 = Path('<DEV_EVAL_ROOT>/jh_dev/program_mapping_nlp_v1/harness_v2')
EMB_MODEL = Path('<DEV_EVAL_ROOT>/jh_dev/govparse-mvp-windows/govparse-mvp/data/semantic_budget_recall_20261007_v1/models/Qwen__Qwen3-Embedding-8B')
PROT = Path('<DEV_EVAL_ROOT>/jh_dev/govparse-mvp-windows/govparse-mvp/docs/experiments/a1-a7/chunk_optimization_harness_v2/reranker_v1/protocol.json')
VN_PARAMS = dict(bv=1.0, bs=0.75, k1=1.5)
TOPK = 30

T = json.loads((RUN / 'texts.json').read_text(encoding='utf-8'))
TOK = json.loads((RUN / 'kiwi_tokens.json').read_text(encoding='utf-8'))
keys = T['keys']; kix = {k: i for i, k in enumerate(keys)}; N = len(keys)
sha = lambda x: hashlib.sha256(x.encode('utf-8')).hexdigest()  # noqa: E731
TYPES = ['purpose_content', 'beneficiary', 'legal']
CONDS = {'N_name': ['name'], 'A_all': ['all'], 'B_split': TYPES[:2], 'C_split_legal': TYPES,
         'D_nam_all': ['nam_all'], 'D_nam_content': ['nam_content'], 'D_nam_content_nomerge': ['nam_content_nomerge']}
queries = [(g, k, [o for o in ks if o != k]) for g, ks in T['eval_groups'].items() for k in ks]
qix = [kix[k] for _, k, _ in queries]


def texts_of(t):
    return [T['texts'][t][k] for k in keys]


def has(t):
    return np.array([T['has'][t][k] for k in keys]) if t in TYPES else np.ones(N, bool)


# ---------------- VN BM25 (vectorized exp_M05_VN.vn_field, single field) ----------------
def vn_parts(t=None, toks=None):
    toks = toks if toks is not None else [TOK[sha(x)] for x in texts_of(t)]
    N = len(toks)
    vocab = {}
    rows, cols, vals = [], [], []
    for i, d in enumerate(toks):
        for w, n in Counter(d).items():
            rows.append(i); cols.append(vocab.setdefault(w, len(vocab))); vals.append(n)
    tf = sp.csr_matrix((np.array(vals, np.float64), (rows, cols)), shape=(N, len(vocab)))
    V = np.array([len(d) / max(1, len(set(d))) for d in toks]); S = np.array([len(set(d)) for d in toks], float)
    aV, aS = max(1e-9, V.mean()), max(1e-9, S.mean())
    df = np.asarray((tf > 0).sum(0)).ravel()
    idf = np.log(1 + (N - df + 0.5) / (df + 0.5))
    c = tf.tocoo()
    x = c.data / ((1 - VN_PARAMS['bv']) + VN_PARAMS['bv'] * V[c.row] / aV) / ((1 - VN_PARAMS['bs']) + VN_PARAMS['bs'] * S[c.row] / aS)
    sat = sp.csr_matrix((x / (VN_PARAMS['k1'] + x), (c.row, c.col)), shape=tf.shape)   # tt/(k1+tt), single field
    qw = tf.multiply(idf[None, :]).tocsr()                                              # qf * idf
    return qw, sat, toks, vocab


def vn_scores(t):
    """cached per (type, text-set hash) so the CPU-only pass (VN_ONLY=1) can run while the GPU embeds"""
    tag = hashlib.sha256(''.join(sha(x) for x in texts_of(t)).encode()).hexdigest()[:16]
    cf = RUN / 'vn_cache' / f'vn_{t}_{tag}.npz'
    if cf.exists():
        z = np.load(cf)
        return z['fwd'], z['sym']
    qw, sat, _, _ = vn_parts(t)
    fwd = np.asarray((qw[qix] @ sat.T).todense())          # q -> c
    bwd = np.asarray((qw @ sat[qix].T).todense()).T        # c -> q (candidate as query, query as document)
    cf.parent.mkdir(exist_ok=True)
    tmp = cf.with_name(cf.stem + '.tmp.npz'); np.savez(tmp, fwd=fwd, sym=0.5 * (fwd + bwd)); tmp.replace(cf)
    return fwd, 0.5 * (fwd + bwd)


def vn_check():
    """the vectorized form equals the exec'd original vn_field on a 300-candidate sample (single field)"""
    src = (HV2 / 'experiments' / 'exp_M05_VN.py').read_text(encoding='utf-8')
    keep = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name in ('_vs', 'vn_field')]
    rng = np.random.default_rng(0)
    sample = sorted(set(rng.choice(N, 300, replace=False).tolist()) | set(qix))
    toks = [TOK[sha(texts_of('all')[i])] for i in sample]

    class FT:  # shim: one field 'text'
        FIELDS = ['text']
        @staticmethod
        def field_tokens(ctx, i):
            return {'text': toks[i]}

    g = {'np': np, 'ft': FT, 'math': math, 'Counter': Counter}
    for n in keep:
        exec(compile(ast.get_source_segment(src, n), 'exp_M05_VN.py', 'exec'), g)
    sq = [sample.index(q) for q in qix]
    ctx = type('C', (), {'qi': sq, 'ci': list(range(len(sample)))})()
    orig = g['vn_field'](ctx, **VN_PARAMS)
    qw, sat, _, _ = vn_parts(toks=toks)  # vectorized on the same sample (corpus stats over the sample, as the original)
    fwd = np.asarray((qw[sq] @ sat.T).todense())
    d = float(np.abs(orig - fwd).max())
    assert d < 1e-6, f'vectorized VN differs from exp_M05_VN.vn_field: {d}'
    return d


# ---------------- Qwen embedding ----------------
_emb = None


def qwen_E(t):
    global _emb
    texts = texts_of(t); hs = [sha(x) for x in texts]
    store = RUN / 'emb_cache' / f'qwen_{t}.npz'
    store.parent.mkdir(exist_ok=True)
    have = {}
    for f in sorted(store.parent.glob('qwen_*.npz')):  # one embedding per text, whichever type stored it
        if '.tmp' not in f.name:
            z = np.load(f); have.update(zip(z['h'].tolist(), z['v']))
    todo = {h for h in hs if h not in have}
    if todo:
        import torch
        from sentence_transformers import SentenceTransformer
        if _emb is None:
            _emb = SentenceTransformer(str(EMB_MODEL), device='cuda', local_files_only=True,
                                       model_kwargs={'torch_dtype': torch.bfloat16, 'attn_implementation': 'sdpa'},
                                       tokenizer_kwargs={'padding_side': 'left'})
            _emb.max_seq_length = 2048
        by = dict(zip(hs, texts))
        todo = sorted(todo, key=lambda h: len(by[h]))
        for s in range(0, len(todo), 4096):
            part = todo[s:s + 4096]
            # long texts with batch 64 filled the 96 GB GPU and spilled into shared memory (run stalled 2026-10-28 17:36);
            # batch shrinks with text length (texts are sorted short->long)
            bs = 64 if len(by[part[-1]]) < 1500 else 16 if len(by[part[-1]]) < 4000 else 8
            v = _emb.encode([by[h] for h in part], batch_size=bs, normalize_embeddings=True, convert_to_numpy=True)
            have.update(zip(part, v.astype(np.float16)))
            tmp = store.with_name(store.stem + '.tmp.npz')  # atomic: a kill mid-write never corrupts the cache
            own = [h for h in dict.fromkeys(hs) if h in have]  # store only this type's texts (no copies of other types)
            np.savez(tmp, h=np.array(own), v=np.stack([have[h] for h in own])); tmp.replace(store)
            print(f'  qwen {t}: {min(s + 4096, len(todo))}/{len(todo)} (batch {bs})', flush=True)
    return np.stack([have[h] for h in hs]).astype(np.float32)


# ---------------- Reranker (verbatim core of score_reranker_full.py) ----------------
_rr = None
RR_CACHE = RUN / 'reranker_cache.json'
rr_done = json.loads(RR_CACHE.read_text(encoding='utf-8')) if RR_CACHE.exists() else {}


def rerank(qtext, dtext):
    global _rr
    key = sha(qtext) + '|' + sha(dtext)
    if key in rr_done:
        return rr_done[key]
    if _rr is None:
        import torch
        torch.backends.cuda.enable_flash_sdp(True); torch.backends.cuda.enable_mem_efficient_sdp(True); torch.backends.cuda.enable_math_sdp(False)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from huggingface_hub import snapshot_download
        prot = json.loads(PROT.read_text(encoding='utf-8'))
        local = snapshot_download(prot['model'], revision=prot['revision'], local_files_only=True)
        tok = AutoTokenizer.from_pretrained(local, padding_side='left')
        model = AutoModelForCausalLM.from_pretrained(local, torch_dtype=torch.bfloat16, attn_implementation='sdpa').cuda().eval()
        prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        no, yes = [tok.convert_tokens_to_ids(x) for x in ['no', 'yes']]
        _rr = (torch, tok, model, prefix, suffix, no, yes, prot)
    torch, tok, model, prefix, suffix, no, yes, prot = _rr
    text = prefix + f"<Instruct>: {prot['instruction']}\n<Query>: {qtext}\n<Document>: {dtext}" + suffix
    inp = tok(text, add_special_tokens=False, truncation=False, return_tensors='pt').to('cuda'); assert inp.input_ids.shape[1] <= 32768
    with torch.inference_mode():
        s = torch.softmax(model(**inp, logits_to_keep=1, use_cache=False).logits[0, -1, [no, yes]].float(), dim=-1)[1].item()
    rr_done[key] = s
    if len(rr_done) % 200 == 0:
        save_rr()
    return s


def save_rr():
    tmp = RR_CACHE.with_suffix('.tmp'); tmp.write_text(json.dumps(rr_done), encoding='utf-8'); tmp.replace(RR_CACHE)


# ---------------- assembly ----------------
def combine(per_type, types, fallback):
    if len(types) == 1:
        return per_type[types[0]]
    num = np.zeros_like(fallback); den = np.zeros_like(fallback)
    for t in types:
        hv = has(t); both = hv[qix][:, None] & hv[None, :]
        num += np.where(both, per_type[t], 0); den += both
    return np.where(den > 0, num / np.maximum(den, 1), fallback)


def order(row, qi):
    s = row.copy(); s[qi] = -np.inf
    return list(np.lexsort((np.arange(N), -s)))[:-1]  # score desc, ties by pool index; query (last) dropped


def rrf_rank(*orders):
    sc = np.zeros(N)
    for o in orders:
        sc[np.array(o)] += 1 / (60 + np.arange(1, len(o) + 1))
    return sc


print('vn vectorized vs original max|diff| =', vn_check(), flush=True)
all_types = sorted({t for ts in CONDS.values() for t in ts} | {'name'})
if os.environ.get('VN_ONLY') == '1':
    from multiprocessing.pool import ThreadPool
    with ThreadPool(8) as tp:  # scipy sparse products release the GIL
        tp.map(vn_scores, all_types)
    print('VN cached for', all_types)
    sys.exit(0)
if os.environ.get('EMB_ONLY') == '1':
    for t in ['all', 'purpose_content', 'beneficiary', 'legal'] + [t for t in all_types if t.startswith('nam_')]:
        qwen_E(t)
    print('embeddings cached'); sys.exit(0)
VNF, VNS, QW, E = {}, {}, {}, {}
for t in all_types:
    print('stage1 features', t, flush=True)
    VNF[t], VNS[t] = vn_scores(t)
    E[t] = qwen_E(t)
    QW[t] = E[t][qix] @ E[t].T

rows = []
for cond, types in CONDS.items():
    fb = 'name'
    S1 = {'qwen': combine(QW, types, QW[fb]), 'vn': combine(VNF, types, VNF[fb]), 'vn_sym': combine(VNS, types, VNS[fb])}
    rtext = (lambda k: T['texts'][types[0]][k]) if len(types) == 1 else (lambda k: T['texts']['all'][k])  # reranker sees one text per program
    for a, (g, k, pos) in enumerate(queries):
        qi = kix[k]
        base_orders = {n: order(S1[n][a], qi) for n in S1}
        base_orders['qwen+vn'] = order(rrf_rank(base_orders['qwen'], base_orders['vn']), qi)
        base_orders['qwen+vn_sym'] = order(rrf_rank(base_orders['qwen'], base_orders['vn_sym']), qi)
        for b, o in base_orders.items():
            top = o[:TOPK]; br = {j: r for r, j in enumerate(o, 1)}
            rr_s = {j: rerank(rtext(k), rtext(keys[j])) for j in top}
            rr = {j: r for r, j in enumerate(sorted(top, key=lambda j: (-rr_s[j], br[j])), 1)}

            def reverse_rank(j):  # rank of the query among candidate j's Qwen neighbours (universe = pool incl. query)
                if len(types) == 1:
                    sims = E[types[0]] @ E[types[0]][j]
                else:
                    num = np.zeros(N); den = np.zeros(N)
                    for t in types:
                        hv = has(t); both = hv & hv[j]
                        num += np.where(both, E[t] @ E[t][j], 0); den += both
                    sims = np.where(den > 0, num / np.maximum(den, 1), E[fb] @ E[fb][j])
                sims[j] = -np.inf
                return 1 + int((sims > sims[qi]).sum())

            def resort(seq, score):
                return sorted(seq, key=lambda j: (-score[j], br[j]))

            rrf60 = resort(top, {j: 1 / (60 + br[j]) + 1 / (60 + rr[j]) for j in top}) + o[TOPK:]
            recip_of = lambda oo: (lambda b2: resort(oo[:TOPK], {j: 0.8 / (60 + b2[j]) + 0.2 / (60 + reverse_rank(j)) for j in oo[:TOPK]}) + oo[TOPK:])({j: r for r, j in enumerate(oo, 1)})  # noqa: E731
            finals = {'none': o, 'rrf60_reranker': rrf60, 'reciprocal': recip_of(o), 'rrf60_reranker->reciprocal': recip_of(rrf60)}
            for s2, fo in finals.items():
                pos_rank = {j: r for r, j in enumerate(fo, 1)}
                for p in pos:
                    rows.append({'condition': cond, 'stage1': b, 'stage2': s2, 'config': f'{b}|{s2}', 'group': g,
                                 'query': T['names'][k], 'positive': T['names'][p], 'rank': pos_rank[kix[p]]})
    save_rr()
    print('done', cond, flush=True)

summary = {}
for r in rows:
    summary.setdefault((r['condition'], r['config']), []).append(r['rank'])
table = [{'condition': c, 'config': cf, 'ranks': rs, 'MRR': float(np.mean([1 / x for x in rs])), 'median': float(np.median(rs)),
          'R@10': float(np.mean([x <= 10 for x in rs])), 'R@30': float(np.mean([x <= 30 for x in rs]))} for (c, cf), rs in summary.items()]
meta = {'pool': N, 'queries': len(queries), 'random_expected_rank': N / 2, 'primary': 'qwen+vn|rrf60_reranker',
        'texts_sha256': hashlib.sha256((RUN / 'texts.json').read_bytes()).hexdigest(),
        'kiwi_sha256': hashlib.sha256((RUN / 'kiwi_tokens.json').read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT / 'pipeline_results.json').write_text(json.dumps({'meta': meta, 'table': table, 'rows': rows}, ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False))
for x in sorted(table, key=lambda x: -x['MRR'])[:25]:
    print(x['condition'], x['config'], x['ranks'], round(x['MRR'], 3))
