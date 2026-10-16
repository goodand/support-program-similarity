"""Offline evaluation; reference labels belong only in this controller step."""
import statistics


def evaluate(rankings, positives, k=30):
    if not isinstance(k, int) or isinstance(k, bool) or k < 1:
        raise ValueError('k must be a positive integer')
    if not positives or set(rankings) != set(positives):
        raise ValueError('Ranking and reference query coverage must match and be nonempty')
    rows = {}
    for qid, relevant in positives.items():
        if not isinstance(relevant, list) or not relevant or any(not isinstance(x, str) for x in relevant):
            raise ValueError('Each reference must be a nonempty list of candidate IDs')
        if len(relevant) != len(set(relevant)):
            raise ValueError('Duplicate reference IDs')
        ranking = rankings[qid]
        if not isinstance(ranking, list) or any(not isinstance(x, str) for x in ranking):
            raise ValueError('Rankings must be lists of candidate IDs')
        if len(ranking) != len(set(ranking)):
            raise ValueError('Duplicate ranked IDs')
        truth = set(relevant)
        ranks = [i for i, cid in enumerate(ranking[:k], 1) if cid in truth]
        rows[qid] = {
            'AP': sum(i / rank for i, rank in enumerate(ranks, 1)) / len(truth),
            'RR': 1 / ranks[0] if ranks else 0,
            'Recall': len(ranks) / len(truth),
            'all_positive_min_k': max(ranks) if len(ranks) == len(truth) else None,
        }
    complete = all(r['all_positive_min_k'] is not None for r in rows.values())
    return {
        'cutoff': k,
        'AP_denominator': 'all known positives, including those missing below cutoff',
        'MAP': statistics.mean(r['AP'] for r in rows.values()),
        'MRR': statistics.mean(r['RR'] for r in rows.values()),
        'macro_recall': statistics.mean(r['Recall'] for r in rows.values()),
        'all_positive_min_k': max(r['all_positive_min_k'] for r in rows.values()) if complete else None,
        'per_query': rows,
    }
