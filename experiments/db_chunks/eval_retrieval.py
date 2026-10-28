"""Retrieval evaluation of chunking conditions (RAG retrieval stage), 2026 central pool from provider A's tabular data.

Conditions: N_name (사업명만), A_all (나오는 텍스트 전부, 1 chunk), B_split (목적·내용 / 수혜대상), C_split_legal (+ 근거법령).
Methods: bm25 (character bigrams, k1=1.2, b=0.75) and qwen (Qwen3-Embedding-8B cosine, no instruction, symmetric).
Multi-chunk conditions: per-type similarity between same-type chunks; a type counts only when both programs have that
field (otherwise the chunk is just the name); combined score = mean over counted types (fallback: name-only score).
Queries = programs of the evaluable groups in eval_set_exact_2026.json; positive = the other matched program(s) of the group.
Rank = 1 + #candidates with a higher score + (#ties)/2 (tie-averaged); the query itself is removed from the pool.
Embeddings are cached per (text sha256) under runs/<run>/emb_cache/ so runs resume.
Usage: <DEV_EVAL_ROOT>/_embedding_v3_env/Scripts/python.exe -I -X utf8 eval_retrieval.py <run> <out_name> [bm25|qwen|all]
"""
import hashlib
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.stdout.reconfigure(encoding='utf-8')
HERE = Path(__file__).resolve().parent
RUN = HERE / 'runs' / sys.argv[1]
OUT = RUN / sys.argv[2]
METHODS = ['bm25', 'qwen'] if len(sys.argv) < 4 or sys.argv[3] == 'all' else [sys.argv[3]]
OUT.mkdir(exist_ok=True)
EXPORT = RUN / 'chunks_export.jsonl'
EXPORT_SHA = '98aa4ba6805be8105d9b4ab36686dcb02e4e3a34c44e8196e6aecf096287e5a6'
assert hashlib.sha256(EXPORT.read_bytes()).hexdigest() == EXPORT_SHA, 'chunks_export.jsonl changed'
MODEL = Path('<DEV_EVAL_ROOT>/jh_dev/govparse-mvp-windows/govparse-mvp/data/semantic_budget_recall_20261007_v1/models/Qwen__Qwen3-Embedding-8B')
TYPES = ['purpose_content', 'beneficiary', 'legal']
CONDS = {'N_name': ['name'], 'A_all': ['all'], 'B_split': TYPES[:2], 'C_split_legal': TYPES}

progs = [json.loads(x) for x in EXPORT.read_text(encoding='utf-8').splitlines()]
keys = [p['key'] for p in progs]
kix = {k: i for i, k in enumerate(keys)}
for p in progs:
    p['name_chunk'] = p['name']
ev = json.loads((RUN / 'eval_set_exact_2026.json').read_text(encoding='utf-8'))
groups = {}
for m in ev['matched_programs']:
    if m['그룹ID'] in ev['groups_evaluable']:
        groups.setdefault(m['그룹ID'], []).append(m['keys'][0])
queries = [(g, k, [o for o in ks if o != k]) for g, ks in groups.items() for k in ks]
missing = [k for _, k, _ in queries if k not in kix]
assert not missing, f'eval programs missing from pool: {missing}'
qix = [kix[k] for _, k, _ in queries]


# ---------- 평가자 B 자료 청크 (사용자 요청 2026-10-28): 평가 그룹 사업만 평가자 B 엑셀 행 텍스트로 바꾼다 ----------
EVAL_B = Path('<DEV_EVAL_ROOT>/jh_dev/govparse-claude-lab/outputs/goldenset_similar_groups_xlsx/<평가그룹_엑셀>.xlsx')
EVAL_B_SHA = 'f185d861455f2e9fee9fe35dd517452cdaf342d653ec7d7ffa7432172f32aee2'
EVAL_B_BUDGET_COLS = {'26예산(억원)', '예산서_2026금액(백만원)', '원문대조_금액'}
EVAL_B_CONTENT_COLS = ['사업명', '소관부처', '사업목적', '지원대상(업종)', '지원내용', '지원조건', '지원방식', '특이사항']
assert hashlib.sha256(EVAL_B.read_bytes()).hexdigest() == EVAL_B_SHA, 'evaluator-B workbook changed'
import openpyxl  # noqa: E402
_ws = openpyxl.load_workbook(EVAL_B, read_only=True)['사업목록']
_it = _ws.iter_rows(values_only=True)
_h = list(next(_it))
evalb_rows = {}
eval_names = {m['keys'][0]: m['사업명'] for m in ev['matched_programs'] if m['그룹ID'] in ev['groups_evaluable']}
for r in _it:
    d = dict(zip(_h, r))
    for k, nm in eval_names.items():
        if d['사업명'] == nm:
            evalb_rows[k] = d
assert set(evalb_rows) == set(eval_names), 'evaluator B rows not found for every eval program'
OVERRIDE = {
    # D_evalb_all: every column except budget columns (user's literal rule; includes group ids/names → leakage expected)
    'evalb_all': {k: ' / '.join(f'{c}: {v}' for c, v in d.items() if c not in EVAL_B_BUDGET_COLS and v not in (None, ''))
                for k, d in evalb_rows.items()},
    # D_evalb_content: program-description columns only (control: no group id/name, no efficiency proposal, no mapping columns)
    'evalb_content': {k: ' / '.join(f'{c}: {d[c]}' for c in EVAL_B_CONTENT_COLS if d.get(c) not in (None, ''))
                    for k, d in evalb_rows.items()},
}
CONDS.update({'D_evalb_all': ['evalb_all'], 'D_evalb_content': ['evalb_content']})


def texts_of(t):
    if t in OVERRIDE:  # eval programs get evaluator B text; every other program keeps its provider-A A_all chunk
        return [OVERRIDE[t].get(p['key'], p['all']) for p in progs]
    return [p['name_chunk'] if t == 'name' else p[t] for p in progs]


def has(t):
    if t in ('name', 'all') or t in OVERRIDE:
        return np.ones(len(progs), bool)
    return np.array([p['has'][t] for p in progs])


# ---------- bm25 ----------
def bigrams(s):
    s = ''.join(ch for ch in s.lower() if not ch.isspace())
    return [s[i:i + 2] for i in range(len(s) - 1)] or [s]


def bm25_matrix(texts, k1=1.2, b=0.75):
    vocab, rows, cols, vals = {}, [], [], []
    for i, t in enumerate(texts):
        for g, n in Counter(bigrams(t)).items():
            rows.append(i); cols.append(vocab.setdefault(g, len(vocab))); vals.append(n)
    tf = sp.csr_matrix((vals, (rows, cols)), shape=(len(texts), len(vocab)), dtype=np.float32)
    dl = np.asarray(tf.sum(1)).ravel(); avg = dl.mean()
    df = np.asarray((tf > 0).sum(0)).ravel()
    idf = np.log(1 + (len(texts) - df + 0.5) / (df + 0.5)).astype(np.float32)
    tf = tf.tocoo()
    w = tf.data * (k1 + 1) / (tf.data + k1 * (1 - b + b * dl[tf.row] / avg))
    doc = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape)
    qbin = sp.csr_matrix((np.ones_like(tf.data), (tf.row, tf.col)), shape=tf.shape).multiply(idf[None, :]).tocsr()
    return qbin, doc


def bm25_scores(t):
    qbin, doc = bm25_matrix(texts_of(t))
    return np.asarray((qbin[qix] @ doc.T).todense(), dtype=np.float64)  # (n_queries, n_pool)


# ---------- qwen ----------
def qwen_scores(t):
    texts = texts_of(t)
    cache = RUN / 'emb_cache'
    cache.mkdir(exist_ok=True)
    h = [hashlib.sha256(x.encode('utf-8')).hexdigest() for x in texts]
    uniq = sorted(set(h))
    store = cache / f'qwen_{t}.npz'
    have = {}
    if store.exists():
        z = np.load(store)
        have = dict(zip(z['h'].tolist(), z['v']))
    todo = [x for x in uniq if x not in have]
    if todo:
        import torch
        from sentence_transformers import SentenceTransformer
        global _model
        if '_model' not in globals():
            _model = SentenceTransformer(str(MODEL), device='cuda', local_files_only=True,
                                         model_kwargs={'torch_dtype': torch.bfloat16, 'attn_implementation': 'sdpa'},
                                         tokenizer_kwargs={'padding_side': 'left'})
            _model.max_seq_length = 2048
        by_h = {hh: x for hh, x in zip(h, texts)}
        order = sorted(todo, key=lambda x: len(by_h[x]))
        for s in range(0, len(order), 4096):
            part = order[s:s + 4096]
            v = _model.encode([by_h[x] for x in part], batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
            have.update(zip(part, v.astype(np.float16)))
            np.savez(store, h=np.array(list(have)), v=np.stack(list(have.values())))
            print(f'  qwen {t}: {min(s + 4096, len(order))}/{len(order)}', flush=True)
    E = np.stack([have[x] for x in h]).astype(np.float32)
    return E[qix] @ E.T


def rank_of(scores, qi, pos):
    s = scores.copy()
    s[qi] = -np.inf
    out = []
    for pj in pos:
        v = s[pj]
        out.append(1 + int((s > v).sum()) + ((s == v).sum() - 1) / 2)
    return out


results = []
for method in METHODS:
    fn = bm25_scores if method == 'bm25' else qwen_scores
    per_type = {}
    for t in ['name', 'all'] + TYPES + list(OVERRIDE):
        print(method, t, flush=True)
        per_type[t] = fn(t)
    for cond, types in CONDS.items():
        if len(types) == 1:
            S = per_type[types[0]]
        else:
            num = np.zeros_like(per_type['name']); den = np.zeros_like(per_type['name'])
            for t in types:
                hv = has(t)
                both = hv[qix][:, None] & hv[None, :]
                num += np.where(both, per_type[t], 0); den += both
            S = np.where(den > 0, num / np.maximum(den, 1), per_type['name'])
        for qn, (g, k, pos) in enumerate(queries):
            r = rank_of(S[qn], kix[k], [kix[o] for o in pos])
            results.append({'method': method, 'condition': cond, 'group': g, 'query': progs[kix[k]]['name'], 'query_key': k,
                            'positives': [progs[kix[o]]['name'] for o in pos], 'rank': r[0] if len(r) == 1 else r,
                            'query_fields': {t: bool(progs[kix[k]]['has'][t]) for t in TYPES},
                            'positive_fields': [{t: bool(progs[kix[o]]['has'][t]) for t in TYPES} for o in pos]})

N = len(progs) - 1
summary = {}
for method in METHODS:
    for cond in CONDS:
        rs = [x['rank'] for x in results if x['method'] == method and x['condition'] == cond]
        summary[f'{method}|{cond}'] = {'ranks': rs, 'median_rank': float(np.median(rs)), 'MRR': float(np.mean([1 / r for r in rs])),
                                       **{f'R@{k}': float(np.mean([r <= k for r in rs])) for k in (5, 10, 30, 100)}}
meta = {'pool': len(progs), 'queries': len(queries), 'random_expected_rank': (N + 1) / 2, 'export_sha256': EXPORT_SHA,
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(OUT / f'results_{"_".join(METHODS)}.json').write_text(json.dumps({'meta': meta, 'summary': summary, 'rows': results},
                                                                   ensure_ascii=False, indent=1), encoding='utf-8')
print(json.dumps(meta, ensure_ascii=False))
for k, v in summary.items():
    print(k, 'ranks', v['ranks'], 'MRR', round(v['MRR'], 3))
