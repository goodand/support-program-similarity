"""Explicit offline scoring CLI. This process may read private reference labels."""
import argparse
import hashlib
import json
from pathlib import Path
from .evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rankings', type=Path, required=True)
    parser.add_argument('--positives', type=Path, required=True)
    parser.add_argument('--k', type=int, default=30)
    args = parser.parse_args()
    # Freeze the exact bytes that will be evaluated before reading references.
    raw = args.rankings.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rankings = json.loads(raw)
    reference_raw = args.positives.read_bytes()
    gold = json.loads(reference_raw)
    result = evaluate(rankings, gold, args.k)
    result['prediction_sha256'] = digest
    result['reference_sha256'] = hashlib.sha256(reference_raw).hexdigest()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
