"""Task-only generative ranking with local aliases and bounded quote repair."""
import argparse
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import platform


DEFAULT_MODEL = 'gpt-6.1-sol'
DEFAULT_EFFORT = 'low'
MAX_QUOTE_REPAIRS = 2
EVIDENCE_KEYS = {'candidate_id', 'query_quote', 'candidate_quote', 'reason'}
RESULT_KEYS = {'query_id', 'ranked_candidate_ids', 'top5_evidence', 'input_issues'}


@dataclass(frozen=True)
class PreparedCorpus:
    task: dict
    query_ids: tuple
    allowed: dict
    query_original_ids: dict
    candidate_original_ids: dict
    top_k: int


@dataclass(frozen=True)
class QuoteIssue:
    result_index: int
    evidence_index: int
    field: str
    query_id: str
    candidate_id: str

    @property
    def message(self):
        return (f'{self.query_id}/{self.candidate_id}: {self.field} must be '
                'a literal substring of the corresponding text, 1..150 characters')


class PredictionValidationError(ValueError):
    def __init__(self, errors):
        self.errors = tuple(errors)
        super().__init__('; '.join(self.errors))


class QuoteValidationError(PredictionValidationError):
    def __init__(self, issues):
        self.issues = tuple(issues)
        super().__init__(issue.message for issue in self.issues)


def build_schema():
    """Keep the historical experiment's schema; validate counts in Python."""
    quote = {'type': 'string'}
    evidence = {
        'type': 'object',
        'properties': {key: quote for key in
                       ['candidate_id', 'query_quote', 'candidate_quote', 'reason']},
        'required': ['candidate_id', 'query_quote', 'candidate_quote', 'reason'],
        'additionalProperties': False,
    }
    item = {
        'type': 'object',
        'properties': {
            'query_id': {'type': 'string'},
            'ranked_candidate_ids': {'type': 'array', 'items': {'type': 'string'}},
            'top5_evidence': {'type': 'array', 'items': evidence},
            'input_issues': {'type': 'array', 'items': {'type': 'string'}},
        },
        'required': ['query_id', 'ranked_candidate_ids', 'top5_evidence', 'input_issues'],
        'additionalProperties': False,
    }
    return {
        'type': 'object',
        'properties': {'results': {'type': 'array', 'items': item},
                       'completed': {'type': 'boolean'}},
        'required': ['results', 'completed'],
        'additionalProperties': False,
    }


def prepare_corpus(rows, top_k=30):
    """Allowlist model-visible fields; retain original IDs only in local maps."""
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError('top_k must be a positive integer')
    if not isinstance(rows, list) or not rows:
        raise ValueError('Corpus must be a nonempty list')
    seen = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f'Corpus row {index} must be an object')
        identifier = row.get('id')
        if not isinstance(identifier, str) or not identifier.strip() or identifier in seen:
            raise ValueError('Corpus IDs must be nonempty, globally unique strings')
        seen.add(identifier)
        if row.get('role') not in ('query', 'candidate'):
            raise ValueError(f'Corpus row {index} needs role query/candidate')
        if not isinstance(row.get('text'), str) or not row['text'].strip():
            raise ValueError(f'Corpus row {index} needs nonempty text')
        if 'name' in row and not isinstance(row['name'], str):
            raise ValueError(f'Corpus row {index} name must be a string when provided')
        exclusions = row.get('excluded_candidate_ids', [])
        if not isinstance(exclusions, list) or any(not isinstance(x, str) for x in exclusions):
            raise ValueError('Exclusions must be a list of candidate IDs')
    queries = [row for row in rows if row['role'] == 'query']
    candidates = [row for row in rows if row['role'] == 'candidate']
    if not queries or not candidates:
        raise ValueError('Both queries and candidates are required')
    query_map = {f'Q{i:02d}': row['id'] for i, row in enumerate(queries, 1)}
    candidate_map = {f'C{i:03d}': row['id'] for i, row in enumerate(candidates, 1)}
    candidate_aliases = {original: alias for alias, original in candidate_map.items()}

    def document(key, alias, row):
        result = {key: alias}
        if 'name' in row:
            result['name'] = row['name']
        result['text'] = row['text']
        return result

    task = {'queries': [], 'candidates': [
        document('candidate_id', alias, row)
        for alias, row in zip(candidate_map, candidates)
    ]}
    allowed = {}
    for alias, row in zip(query_map, queries):
        excluded = set(row.get('excluded_candidate_ids', []))
        if excluded - set(candidate_aliases):
            raise ValueError(f'{alias} has an exclusion that is not a candidate ID')
        excluded.update(candidate['id'] for candidate in candidates
                        if candidate['id'] == row['id'] or candidate['text'] == row['text'])
        record = document('query_id', alias, row)
        record['excluded_candidate_ids'] = [candidate_aliases[candidate['id']]
                                            for candidate in candidates
                                            if candidate['id'] in excluded]
        allowed[alias] = frozenset(candidate_map) - set(record['excluded_candidate_ids'])
        if len(allowed[alias]) < top_k:
            raise ValueError(f'{alias} has fewer allowed candidates than top_k')
        task['queries'].append(record)
    return PreparedCorpus(task, tuple(query_map), allowed, query_map, candidate_map, top_k)


def build_prompt(prepared):
    k, evidence_count = prepared.top_k, min(5, prepared.top_k)
    instruction = (
        'Goal: 제공된 예산서 사업 문서만 근거로 각 질의 사업과 지원목적·지원활동·수혜대상·지원조건이 유사한 후보 사업을 찾아 유사성이 높은 순서로 정렬한다.\n'
        f'출력 형식: 지정된 JSON Schema를 채운다. 각 query_id에 ranked_candidate_ids를 유사성 내림차순으로 정확히 {k}개 기록한다. '
        f'top5_evidence에는 상위 {evidence_count}개 후보 각각의 candidate_id, 질의 원문의 연속 부분인 query_quote, 후보 원문의 연속 부분인 candidate_quote, '
        '짧은 유사성 근거 reason을 기록한다. 인용은 각각 150자 이하로 한다. 문서 부족·오염 등 판단 한계는 input_issues에 기록한다.\n'
        f'End 종료 조건: {len(prepared.query_ids)}개 질의를 모두 처리하고, 각 순위에 실제 존재하는 후보 {k}개가 중복 없이 포함되며, '
        f'excluded_candidate_ids를 제외했고, 상위 {evidence_count}개 근거를 원문으로 뒷받침했으면 completed=true로 종료한다.'
    )
    return instruction + '\n\n' + json.dumps(prepared.task, ensure_ascii=False)


def validate_prediction(prediction, prepared):
    """Reject structural failures; expose repair slots only for literal quotes."""
    if not isinstance(prediction, dict) or set(prediction) != {'results', 'completed'}:
        raise PredictionValidationError(['Prediction must match the exact output schema'])
    if prediction['completed'] is not True or not isinstance(prediction['results'], list):
        raise PredictionValidationError(['Prediction must be completed with a results array'])
    results = prediction['results']
    errors, quote_issues, seen = [], [], set()
    query_text = {row['query_id']: row['text'] for row in prepared.task['queries']}
    candidate_text = {row['candidate_id']: row['text'] for row in prepared.task['candidates']}
    if len(results) != len(prepared.query_ids):
        errors.append('Each query must occur exactly once')
    for result_index, result in enumerate(results):
        if not isinstance(result, dict) or set(result) != RESULT_KEYS:
            errors.append(f'Result {result_index} must match the exact result schema')
            continue
        qid = result['query_id']
        if not isinstance(qid, str) or qid not in prepared.allowed or qid in seen:
            errors.append(f'Result {result_index} has an unknown or duplicate query ID')
            continue
        seen.add(qid)
        ranking = result['ranked_candidate_ids']
        valid_ranking = (isinstance(ranking, list) and len(ranking) == prepared.top_k
                         and all(isinstance(cid, str) for cid in ranking))
        if valid_ranking:
            valid_ranking = (len(set(ranking)) == len(ranking)
                             and set(ranking) <= prepared.allowed[qid])
        if not valid_ranking:
            errors.append(f'{qid} must rank exactly top_k distinct allowed candidate IDs')
        input_issues = result['input_issues']
        if not isinstance(input_issues, list) or any(not isinstance(x, str) for x in input_issues):
            errors.append(f'{qid} input_issues must be an array of strings')
        evidence = result['top5_evidence']
        if not isinstance(evidence, list) or len(evidence) != min(5, prepared.top_k):
            errors.append(f'{qid} must supply evidence for the first min(5, top_k) candidates')
            continue
        for evidence_index, item in enumerate(evidence):
            if (not isinstance(item, dict) or set(item) != EVIDENCE_KEYS
                    or any(not isinstance(value, str) for value in item.values())):
                errors.append(f'{qid} evidence {evidence_index} must match the exact evidence schema')
                continue
            cid = item['candidate_id']
            if (not valid_ranking or cid != ranking[evidence_index]
                    or cid not in candidate_text):
                errors.append(f'{qid} evidence must follow the ranked candidate order')
                continue
            if not item['reason'].strip():
                errors.append(f'{qid} evidence must include a nonempty reason')
            for field, source in [('query_quote', query_text[qid]),
                                  ('candidate_quote', candidate_text[cid])]:
                quote = item[field]
                if not 1 <= len(quote) <= 150 or quote not in source:
                    quote_issues.append(QuoteIssue(result_index, evidence_index, field, qid, cid))
    if seen != set(prepared.query_ids):
        errors.append('Query coverage must match the corpus exactly')
    if errors:
        raise PredictionValidationError(errors)
    if quote_issues:
        raise QuoteValidationError(quote_issues)
    return prediction


def build_quote_repair_prompt(previous, prepared, issues):
    affected_queries = {issue.query_id for issue in issues}
    affected_candidates = {issue.candidate_id for issue in issues}
    repair = {'previous_output': previous,
              'validation_errors': [issue.message for issue in issues]}
    return (
        'Goal: 기존 결과의 인용 형식 오류만 원문으로 수정한다.\n'
        '출력 형식: 기존 전체 JSON을 같은 Schema로 반환한다.\n'
        'End 종료 조건: ranked_candidate_ids와 나머지 값은 유지하고 지적된 인용만 제공된 해당 원문 본문의 연속 부분문자열(150자 이하)로 수정했으면 종료한다.\n'
        + json.dumps({'repair': repair,
                      'affected_queries': [row for row in prepared.task['queries']
                                           if row['query_id'] in affected_queries],
                      'affected_candidates': [row for row in prepared.task['candidates']
                                              if row['candidate_id'] in affected_candidates]},
                     ensure_ascii=False)
    )


def require_quote_only_repair(previous, repaired, issues):
    """Freeze every value except the precise quote fields that failed validation."""
    before, after = copy.deepcopy(previous), copy.deepcopy(repaired)
    try:
        for issue in issues:
            for value in (before, after):
                evidence = value['results'][issue.result_index]['top5_evidence'][issue.evidence_index]
                if issue.field not in evidence:
                    raise KeyError(issue.field)
                evidence[issue.field] = None
    except (KeyError, TypeError, IndexError):
        raise PredictionValidationError(['Quote repair changed the output structure']) from None
    if before != after:
        raise PredictionValidationError(['Quote repair changed rankings or another frozen value'])


def _strict_json(raw):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON object key')
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError('Nonfinite JSON constants are not allowed')

    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def _write_json(path, value):
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as handle:
        handle.write(raw)
    return _digest(raw)


def run(corpus, output, *, model=DEFAULT_MODEL, effort=DEFAULT_EFFORT, top_k=30,
        timeout=2400, codex_executable=None, attempt_runner=None):
    """Read one corpus, run a bounded worker, and publish only validated local outputs.

    attempt_runner supports deterministic offline tests. The default imports the
    task-only Codex worker lazily; no reference-label path is accepted or read.
    """
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
        raise ValueError('timeout must be a positive integer')
    if not isinstance(model, str) or not model.strip() or not isinstance(effort, str) or not effort.strip():
        raise ValueError('model and effort must be nonempty strings')
    raw = Path(corpus).read_bytes()
    prepared = prepare_corpus(_strict_json(raw), top_k)
    schema, prompt = build_schema(), build_prompt(prepared)
    if attempt_runner is None:
        from .codex_worker import run_attempt
        attempt_runner = run_attempt
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'attempts').mkdir()
    mapping = {'queries': prepared.query_original_ids, 'candidates': prepared.candidate_original_ids}
    mapping_hash = _write_json(output / 'id_mapping.json', mapping)
    manifest = {
        'method': 'task_only_generative_topk_quote_repair_v1',
        'model': model, 'reasoning_effort': effort, 'top_k': top_k,
        'evidence_per_query': min(5, top_k), 'timeout_seconds_per_attempt': timeout,
        'queries': len(prepared.query_ids), 'candidates': len(prepared.candidate_original_ids),
        'corpus_sha256': _digest(raw), 'initial_prompt_sha256': _digest(prompt.encode('utf-8')),
        'schema_sha256': _digest(json.dumps(schema, sort_keys=True).encode('utf-8')),
        'python_version': platform.python_version(),
        'code_sha256': {path.name: _digest(path.read_bytes())
                        for path in sorted(Path(__file__).parent.glob('*.py'))},
        'model_input_fields': {
            'queries': ['query_id', 'name (optional)', 'text', 'excluded_candidate_ids'],
            'candidates': ['candidate_id', 'name (optional)', 'text'],
        },
        'reference_label_files_read': False,
        'attempt_prompt_sha256': [],
    }
    previous, repair_issues = None, ()
    try:
        for attempt in range(1, MAX_QUOTE_REPAIRS + 2):
            manifest['attempt_prompt_sha256'].append(_digest(prompt.encode('utf-8')))
            prediction = attempt_runner(
                prompt, copy.deepcopy(schema), output / 'attempts' / f'attempt_{attempt:02d}',
                model=model, effort=effort, timeout=timeout, codex_executable=codex_executable)
            if previous is not None:
                require_quote_only_repair(previous, prediction, repair_issues)
            try:
                validate_prediction(prediction, prepared)
                break
            except QuoteValidationError as exc:
                if attempt > MAX_QUOTE_REPAIRS:
                    raise
                previous, repair_issues = copy.deepcopy(prediction), exc.issues
                prompt = build_quote_repair_prompt(previous, prepared, repair_issues)
        by_query = {result['query_id']: result for result in prediction['results']}
        rankings = {prepared.query_original_ids[qid]: [prepared.candidate_original_ids[cid]
                    for cid in by_query[qid]['ranked_candidate_ids']] for qid in prepared.query_ids}
        manifest['artifacts_sha256'] = {
            'id_mapping.json': mapping_hash,
            'blind_predictions.json': _write_json(output / 'blind_predictions.json', prediction),
            'rankings.json': _write_json(output / 'rankings.json', rankings),
        }
        manifest['status'] = 'completed'
        manifest['attempt_count'] = len(manifest['attempt_prompt_sha256'])
        manifest['quote_repair_attempts'] = manifest['attempt_count'] - 1
        _write_json(output / 'manifest.json', manifest)
        return manifest
    except Exception as exc:
        manifest['status'] = 'failed'
        manifest['error_type'] = type(exc).__name__
        manifest['attempt_count'] = len(manifest['attempt_prompt_sha256'])
        try:
            _write_json(output / 'manifest.json', manifest)
        except OSError:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New local output directory')
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--effort', default=DEFAULT_EFFORT)
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--timeout', type=int, default=2400, help='Seconds per attempt')
    parser.add_argument('--codex-executable')
    args = parser.parse_args()
    manifest = run(args.corpus, args.output, model=args.model, effort=args.effort,
                   top_k=args.top_k, timeout=args.timeout, codex_executable=args.codex_executable)
    print(json.dumps({'status': manifest['status'], 'queries': manifest['queries'],
                      'top_k': manifest['top_k'], 'attempts': manifest['attempt_count']}))


if __name__ == '__main__':
    main()
