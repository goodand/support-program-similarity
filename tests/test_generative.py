"""Synthetic protocol tests; these never invoke Codex or load evaluation labels."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from support_program_similarity.generative import (
    PredictionValidationError, QuoteValidationError, build_prompt,
    prepare_corpus, require_quote_only_repair, run, validate_prediction,
)


def synthetic_corpus():
    return [
        {'id': 'original-q-alpha', 'role': 'query', 'name': 'Synthetic question A',
         'text': '가상 교육 지원.\n장비 제공.', 'excluded_candidate_ids': ['original-c-four'],
         'gold': ['NEVER_SEND_SYNTHETIC_MARKER'],
         'provenance': {'note': 'NEVER_SEND_SYNTHETIC_MARKER'},
         'query_id': 'NEVER_SEND_SYNTHETIC_MARKER'},
        {'id': 'original-q-beta', 'role': 'query', 'text': '가상 장비 지원.'},
        {'id': 'original-c-one', 'role': 'candidate', 'name': 'Synthetic candidate A',
         'text': '가상 교육 지원.', 'labels': ['NEVER_SEND_SYNTHETIC_MARKER']},
        {'id': 'original-c-two', 'role': 'candidate', 'text': '가상 장비 지원.'},
        {'id': 'original-c-three', 'role': 'candidate', 'text': '가상 연구 지원.'},
        {'id': 'original-c-four', 'role': 'candidate', 'text': '제외할 합성 후보.'},
        {'id': 'original-c-five', 'role': 'candidate', 'text': '가상 교육 지원.\n장비 제공.'},
    ]


def valid_prediction(prepared):
    queries = {row['query_id']: row['text'] for row in prepared.task['queries']}
    candidates = {row['candidate_id']: row['text'] for row in prepared.task['candidates']}
    results = []
    for qid in prepared.query_ids:
        ranked = sorted(prepared.allowed[qid])[:prepared.top_k]
        results.append({
            'query_id': qid, 'ranked_candidate_ids': ranked,
            'top5_evidence': [{'candidate_id': cid, 'query_quote': queries[qid],
                              'candidate_quote': candidates[cid], 'reason': '합성 지원활동 비교.'}
                             for cid in ranked[:5]],
            'input_issues': [],
        })
    return {'results': results, 'completed': True}


class GenerativeTests(unittest.TestCase):
    def setUp(self):
        self.rows = synthetic_corpus()
        self.prepared = prepare_corpus(self.rows, top_k=2)
        self.valid = valid_prediction(self.prepared)

    def test_aliases_exclusions_and_only_allowed_input_fields(self):
        task = self.prepared.task
        self.assertEqual([row['query_id'] for row in task['queries']], ['Q01', 'Q02'])
        self.assertEqual([row['candidate_id'] for row in task['candidates']],
                         ['C001', 'C002', 'C003', 'C004', 'C005'])
        self.assertEqual(task['queries'][0]['excluded_candidate_ids'], ['C004', 'C005'])
        self.assertEqual(task['queries'][1]['excluded_candidate_ids'], ['C002'])
        self.assertEqual(task['queries'][0]['name'], self.rows[0]['name'])
        self.assertEqual(task['queries'][0]['text'], self.rows[0]['text'])
        prompt = build_prompt(self.prepared)
        self.assertNotIn('NEVER_SEND_SYNTHETIC_MARKER', prompt)
        for row in self.rows:
            self.assertNotIn(row['id'], prompt)
        for query in task['queries']:
            self.assertLessEqual(set(query), {'query_id', 'name', 'text', 'excluded_candidate_ids'})
        for candidate in task['candidates']:
            self.assertLessEqual(set(candidate), {'candidate_id', 'name', 'text'})

    def test_ambiguous_ids_unknown_exclusions_and_insufficient_candidates_fail(self):
        duplicate = copy.deepcopy(self.rows)
        duplicate[1]['id'] = duplicate[0]['id']
        unknown_exclusion = copy.deepcopy(self.rows)
        unknown_exclusion[0]['excluded_candidate_ids'] = ['missing-candidate']
        for rows, k in [(duplicate, 2), (unknown_exclusion, 2), (self.rows, 4),
                        (self.rows, True), (self.rows, 0)]:
            with self.subTest(k=k, duplicate=rows is duplicate):
                with self.assertRaises(ValueError):
                    prepare_corpus(rows, k)

    def test_exact_query_coverage_and_allowed_candidate_ranking(self):
        missing = copy.deepcopy(self.valid)
        missing['results'].pop()
        duplicate_query = copy.deepcopy(self.valid)
        duplicate_query['results'][1] = copy.deepcopy(duplicate_query['results'][0])
        unknown_query = copy.deepcopy(self.valid)
        unknown_query['results'][0]['query_id'] = 'Q99'
        excluded = copy.deepcopy(self.valid)
        excluded['results'][0]['ranked_candidate_ids'][0] = 'C004'
        duplicate_candidate = copy.deepcopy(self.valid)
        duplicate_candidate['results'][0]['ranked_candidate_ids'][1] = 'C001'
        extra_field = copy.deepcopy(self.valid)
        extra_field['results'][0]['extra'] = 'unexpected'
        unfinished = copy.deepcopy(self.valid)
        unfinished['completed'] = False
        integer_completion = copy.deepcopy(self.valid)
        integer_completion['completed'] = 1
        for prediction in [missing, duplicate_query, unknown_query, excluded,
                           duplicate_candidate, extra_field, unfinished, integer_completion]:
            with self.subTest(prediction=prediction):
                with self.assertRaises(PredictionValidationError) as caught:
                    validate_prediction(prediction, self.prepared)
                self.assertNotIsInstance(caught.exception, QuoteValidationError)

    def test_evidence_order_and_count_must_match_the_shortlist(self):
        for change in ('order', 'count'):
            prediction = copy.deepcopy(self.valid)
            evidence = prediction['results'][0]['top5_evidence']
            if change == 'order':
                evidence.reverse()
            else:
                evidence.pop()
            with self.subTest(change=change):
                with self.assertRaises(PredictionValidationError):
                    validate_prediction(prediction, self.prepared)

    def test_quotes_are_literal_and_one_to_150_characters(self):
        rows = copy.deepcopy(self.rows)
        rows[2]['text'] = 'A\r\nB Ａ 가나다' + 'x' * 160
        prepared = prepare_corpus(rows, 2)
        for quote in ['', 'x' * 151, 'A\nB', 'A 가나다', '가']:
            prediction = valid_prediction(prepared)
            prediction['results'][0]['top5_evidence'][0]['candidate_quote'] = quote
            with self.subTest(quote=quote):
                with self.assertRaises(QuoteValidationError):
                    validate_prediction(prediction, prepared)
        prediction = valid_prediction(prepared)
        for result in prediction['results']:
            for evidence in result['top5_evidence']:
                if evidence['candidate_id'] == 'C001':
                    evidence['candidate_quote'] = 'A\r\nB Ａ 가나다'
        validate_prediction(prediction, prepared)

    def test_structural_errors_take_precedence_over_quote_repair(self):
        prediction = copy.deepcopy(self.valid)
        prediction['results'][0]['top5_evidence'][0]['query_quote'] = 'absent'
        prediction['results'][1]['ranked_candidate_ids'] = []
        with self.assertRaises(PredictionValidationError) as caught:
            validate_prediction(prediction, self.prepared)
        self.assertNotIsInstance(caught.exception, QuoteValidationError)

    def test_only_invalid_quote_fields_may_change(self):
        previous = copy.deepcopy(self.valid)
        previous['results'][0]['top5_evidence'][0]['candidate_quote'] = 'absent'
        with self.assertRaises(QuoteValidationError) as caught:
            validate_prediction(previous, self.prepared)
        issues = caught.exception.issues
        require_quote_only_repair(previous, self.valid, issues)
        for change in ('rank', 'reason', 'valid_quote', 'issues'):
            repaired = copy.deepcopy(self.valid)
            result = repaired['results'][0]
            if change == 'rank':
                result['ranked_candidate_ids'].reverse()
                result['top5_evidence'].reverse()
            elif change == 'reason':
                result['top5_evidence'][0]['reason'] = 'changed reason'
            elif change == 'valid_quote':
                result['top5_evidence'][0]['query_quote'] = '교육'
            else:
                result['input_issues'].append('changed issue')
            with self.subTest(change=change):
                with self.assertRaises(PredictionValidationError):
                    require_quote_only_repair(previous, repaired, issues)

    def _execute(self, directory, predictions):
        corpus = directory / 'corpus.synthetic.json'
        corpus.write_text(json.dumps(self.rows, ensure_ascii=False), encoding='utf-8')
        calls = []

        def worker(prompt, schema, output_dir, **options):
            self.assertFalse(output_dir.exists())
            output_dir.mkdir()
            calls.append({'prompt': prompt, 'schema': schema, 'options': options})
            return copy.deepcopy(predictions[min(len(calls) - 1, len(predictions) - 1)])

        return corpus, calls, worker

    def test_successful_repair_restores_original_ids_and_records_private_local_manifest(self):
        invalid = copy.deepcopy(self.valid)
        invalid['results'][0]['top5_evidence'][0]['candidate_quote'] = 'absent'
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            corpus, calls, worker = self._execute(directory, [invalid, self.valid])
            output = directory / 'run'
            manifest = run(corpus, output, top_k=2, attempt_runner=worker)
            self.assertEqual(manifest['status'], 'completed')
            self.assertEqual(manifest['attempt_count'], 2)
            self.assertEqual(manifest['quote_repair_attempts'], 1)
            self.assertEqual(calls[0]['options']['model'], 'gpt-6.1-sol')
            self.assertEqual(calls[0]['options']['effort'], 'low')
            repair_payload = json.loads(calls[1]['prompt'].split('\n', 3)[3])
            self.assertEqual(repair_payload['repair']['previous_output'], invalid)
            self.assertEqual([row['query_id'] for row in repair_payload['affected_queries']], ['Q01'])
            self.assertEqual([row['candidate_id'] for row in repair_payload['affected_candidates']], ['C001'])
            rankings = json.loads((output / 'rankings.json').read_text(encoding='utf-8'))
            self.assertEqual(rankings['original-q-alpha'], ['original-c-one', 'original-c-two'])
            manifest_text = (output / 'manifest.json').read_text(encoding='utf-8')
            for row in self.rows:
                self.assertNotIn(row['id'], manifest_text)
                self.assertNotIn(row['text'], manifest_text)
            self.assertNotIn('NEVER_SEND_SYNTHETIC_MARKER', ''.join(call['prompt'] for call in calls))
            for name, digest in manifest['artifacts_sha256'].items():
                self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), digest)

    def test_nonquote_failure_is_not_retried_or_published(self):
        invalid = copy.deepcopy(self.valid)
        invalid['results'].pop()
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            corpus, calls, worker = self._execute(directory, [invalid])
            output = directory / 'run'
            with self.assertRaises(PredictionValidationError):
                run(corpus, output, top_k=2, attempt_runner=worker)
            self.assertEqual(len(calls), 1)
            self.assertFalse((output / 'rankings.json').exists())
            self.assertFalse((output / 'blind_predictions.json').exists())
            self.assertEqual(json.loads((output / 'manifest.json').read_text())['status'], 'failed')

    def test_quote_repair_is_bounded_to_two_additional_attempts(self):
        invalid = copy.deepcopy(self.valid)
        invalid['results'][0]['top5_evidence'][0]['query_quote'] = 'absent'
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            corpus, calls, worker = self._execute(directory, [invalid])
            with self.assertRaises(QuoteValidationError):
                run(corpus, directory / 'run', top_k=2, attempt_runner=worker)
            self.assertEqual(len(calls), 3)
            self.assertFalse((directory / 'run' / 'rankings.json').exists())

    def test_second_repair_cannot_modify_a_previously_fixed_quote(self):
        first = copy.deepcopy(self.valid)
        first['results'][0]['top5_evidence'][0]['query_quote'] = 'absent'
        first['results'][0]['top5_evidence'][0]['candidate_quote'] = 'absent'
        second = copy.deepcopy(self.valid)
        second['results'][0]['top5_evidence'][0]['candidate_quote'] = 'absent'
        third = copy.deepcopy(self.valid)
        third['results'][0]['top5_evidence'][0]['query_quote'] = '교육'
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            corpus, calls, worker = self._execute(directory, [first, second, third])
            with self.assertRaises(PredictionValidationError):
                run(corpus, directory / 'run', top_k=2, attempt_runner=worker)
            self.assertEqual(len(calls), 3)
            self.assertFalse((directory / 'run' / 'rankings.json').exists())

    def test_existing_output_directory_is_never_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            corpus, calls, worker = self._execute(directory, [self.valid])
            output = directory / 'run'
            output.mkdir()
            sentinel = output / 'keep.txt'
            sentinel.write_text('preserve me', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                run(corpus, output, top_k=2, attempt_runner=worker)
            self.assertEqual(calls, [])
            self.assertEqual(sentinel.read_text(), 'preserve me')
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_invalid_json_is_rejected_before_worker_execution(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            corpus, calls, worker = self._execute(directory, [self.valid])
            for raw in ['[{"id":"a","id":"b"}]', '[NaN]', '{broken']:
                corpus.write_text(raw, encoding='utf-8')
                with self.subTest(raw=raw):
                    with self.assertRaises(ValueError):
                        run(corpus, directory / 'run', top_k=2, attempt_runner=worker)
                    self.assertFalse((directory / 'run').exists())
            self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()
