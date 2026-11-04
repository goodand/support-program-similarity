"""Cosine agreement between two embedding stores (shard folders s*.npz or a single .npz with keys h, v).

Usage: python compare_shards.py <a> <b> <min_cos> [--json out.json]
Exit code 0 when every shared hash has cosine >= min_cos, 1 otherwise (2 when nothing is shared).
"""
import json
import sys
from pathlib import Path

import numpy as np


def load(p):
    p = Path(p)
    files = [p] if p.suffix == '.npz' else sorted(f for f in p.glob('s*.npz') if '.tmp' not in f.name)
    d = {}
    for f in files:
        with np.load(f) as z:
            d.update(zip(z['h'].tolist(), z['v'].astype(np.float32)))
    return d


a, b, thr = load(sys.argv[1]), load(sys.argv[2]), float(sys.argv[3])
hs = sorted(set(a) & set(b))
if not hs:
    print(json.dumps({'n': 0, 'a': sys.argv[1], 'b': sys.argv[2]}))
    sys.exit(2)
A = np.stack([a[h] for h in hs]); B = np.stack([b[h] for h in hs])
A /= np.linalg.norm(A, axis=1, keepdims=True); B /= np.linalg.norm(B, axis=1, keepdims=True)
c = (A * B).sum(1)
res = {'a': sys.argv[1], 'b': sys.argv[2], 'n': len(hs), 'only_a': len(a) - len(hs), 'only_b': len(b) - len(hs),
       'cos_min': float(c.min()), 'cos_mean': float(c.mean()), 'cos_p01': float(np.percentile(c, 1)), 'threshold': thr, 'pass': bool(c.min() >= thr)}
print(json.dumps(res))
if '--json' in sys.argv:
    Path(sys.argv[sys.argv.index('--json') + 1]).write_text(json.dumps(res, indent=1), encoding='utf-8')
sys.exit(0 if res['pass'] else 1)
