"""Qwen embedding retrieval with whole-input window pooling and reranking."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from .windows import text_windows

INSTRUCTION = ('Instruct: Given a government support program, retrieve programs '
               'with similar purposes, beneficiaries, support activities and eligibility conditions.\nQuery: ')


def rank_vectors(rows, vectors):
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or len(vectors) != len(rows) or not np.isfinite(vectors).all():
        raise ValueError('Invalid vector matrix')
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms == 0):
        raise ValueError('Zero vector')
    vectors = vectors / norms[:, None]
    candidates = [i for i, row in enumerate(rows) if row['role'] == 'candidate']
    rankings = {}
    for i, row in enumerate(rows):
        if row['role'] != 'query':
            continue
        excluded = set(row.get('excluded_candidate_ids', []))
        allowed = [j for j in candidates if rows[j]['id'] not in excluded and rows[j]['id'] != row['id'] and rows[j]['text'] != row['text']]
        order = sorted(allowed, key=lambda j: (-float(vectors[i] @ vectors[j]), rows[j]['id']))
        rankings[row['id']] = [rows[j]['id'] for j in order]
    return rankings


def check_rows(rows):
    if not isinstance(rows, list) or not rows:
        raise ValueError('Corpus must be a nonempty list')
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id']:
            raise ValueError('Each row needs a nonempty string id')
        if row.get('role') not in ('query', 'candidate') or not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError('Each row needs role query/candidate and nonempty text')
        exclusions = row.get('excluded_candidate_ids', [])
        if not isinstance(exclusions, list) or any(not isinstance(cid, str) for cid in exclusions):
            raise ValueError('Exclusions must be a list of IDs')
    if len({row['id'] for row in rows}) != len(rows):
        raise ValueError('IDs must be unique')
    if {row['role'] for row in rows} != {'query', 'candidate'}:
        raise ValueError('Both queries and candidates are required')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True, help='Local Qwen3-Embedding-8B snapshot')
    parser.add_argument('--reranker', type=Path, required=True, help='Local Qwen3-Reranker-0.6B snapshot')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    raw = args.corpus.read_bytes()
    rows = json.loads(raw)
    check_rows(rows)
    if not args.model.is_dir() or not args.reranker.is_dir():
        raise ValueError('Local model directories must exist')
    args.output.mkdir(parents=True, exist_ok=False)
    import torch
    from sentence_transformers import SentenceTransformer
    dtype = torch.bfloat16 if args.device.startswith('cuda') else torch.float32
    model = SentenceTransformer(str(args.model), device=args.device, local_files_only=True,
                                model_kwargs={'torch_dtype': dtype})
    model.max_seq_length = 4352
    # Use our frozen domain instruction, never an implicit snapshot default.
    if any('Router' in type(module).__name__ for module in model):
        raise ValueError('Task-routed embedding snapshots are not supported')
    vectors, audit = [], []
    for row in rows:
        text = row['text']
        windows, weights = text_windows(model.tokenizer, text)
        prefix = INSTRUCTION if row['role'] == 'query' else ''
        inputs = [prefix + window for window in windows]
        # Detect hidden tokenizer truncation instead of silently losing text.
        if any(len(model.tokenizer.encode(t)) > model.max_seq_length for t in inputs):
            raise ValueError('Window exceeds model limit after adding instruction')
        encoded = model.encode(inputs, prompt='', batch_size=2, normalize_embeddings=True, show_progress_bar=False)
        vector = np.average(np.asarray(encoded, dtype=np.float32), axis=0, weights=weights)
        norm = np.linalg.norm(vector)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError('Invalid pooled vector')
        vectors.append(vector / norm)
        audit.append({'id': row['id'], 'windows': len(windows), 'tokens': sum(weights),
                      'input_sha256': hashlib.sha256(text.encode()).hexdigest()})
    rankings = rank_vectors(rows, vectors)
    del model
    import gc
    gc.collect()
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    from .reranking import rerank
    rankings = rerank(rows, rankings, args.reranker, args.device)
    (args.output/'rankings.json').write_text(json.dumps(rankings, ensure_ascii=False, indent=2), encoding='utf-8')
    np.save(args.output/'vectors.npy', vectors)
    import importlib.metadata
    manifest = {'corpus_sha256': hashlib.sha256(raw).hexdigest(), 'model_path': str(args.model.resolve()),
                'method': 'whole_input_qwen_then_qwen_reranker_top20_unicode_safe_v1', 'window_tokens': 4096, 'device': args.device,
                'reranker_path': str(args.reranker.resolve()),
                'instruction': INSTRUCTION, 'pooling': 'token weighted mean of normalized windows, then L2',
                'versions': {name: importlib.metadata.version(name) for name in ['numpy', 'torch', 'transformers', 'sentence-transformers']},
                'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
                'inputs': audit}
    (args.output/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
