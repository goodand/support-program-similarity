import unittest
import numpy as np
from support_program_similarity.evaluation import evaluate
from support_program_similarity.retrieval import check_rows, rank_vectors
from support_program_similarity.reranking import reorder
from tools.check_publish import inspect
from support_program_similarity.windows import text_windows


class CoreTests(unittest.TestCase):
    def test_unicode_windows_preserve_original(self):
        class ByteTokenizer:
            def encode(self, text, **kwargs): return list(text.encode('utf-8'))
            def __call__(self, text, **kwargs):
                return {'input_ids': self.encode(text), 'offset_mapping': [(i,i+1) for i,c in enumerate(text) for _ in c.encode('utf-8')]}
        text = '가나다  ABC\n한글'
        chunks, weights = text_windows(ByteTokenizer(), text, 4)
        self.assertEqual(''.join(chunks), text)
        self.assertTrue(all(w <= 4 for w in weights))
        with self.assertRaises(ValueError): text_windows(ByteTokenizer(), '가', 1)

    def test_missing_positive_penalizes_ap(self):
        m = evaluate({'Q': ['a', 'b']}, {'Q': ['a', 'c']}, 2)
        self.assertEqual(m['MAP'], .5)
        self.assertEqual(m['macro_recall'], .5)
        self.assertIsNone(m['all_positive_min_k'])

    def test_full_metrics(self):
        m = evaluate({'Q': ['x', 'a', 'b']}, {'Q': ['a', 'b']}, 3)
        self.assertAlmostEqual(m['MAP'], (1/2+2/3)/2)
        self.assertEqual(m['MRR'], .5)
        self.assertEqual(m['all_positive_min_k'], 3)

    def test_macro_recall(self):
        m = evaluate({'Q': ['a'], 'R': ['z']}, {'Q': ['a', 'b'], 'R': ['z']}, 1)
        self.assertEqual(m['macro_recall'], .75)

    def test_invalid_evaluation(self):
        for rankings, gold in [({'Q': ['a', 'a']}, {'Q': ['a']}), ({'Q': []}, {'Q': []}), ({'Q': []}, {'R': ['a']})]:
            with self.assertRaises(ValueError): evaluate(rankings, gold)

    def test_reranking_tail_preserved(self):
        self.assertEqual(reorder(['a','b','c'], {'a':.2,'b':.8}, 2), ['b','a','c'])
        self.assertEqual(reorder(['b','a','c'], {'a':.5,'b':.5}, 2), ['b','a','c'])

    def test_reranking_invalid_scores(self):
        with self.assertRaises(ValueError): reorder(['a'], {'b':.5})
        with self.assertRaises(ValueError): reorder(['a'], {'a':float('nan')})

    def test_exclusion_and_tie(self):
        rows = [{'id': 'q', 'role': 'query', 'text': 'query', 'excluded_candidate_ids': ['c']},
                *[{'id': x, 'role': 'candidate', 'text': x} for x in ['b', 'a', 'c']],
                {'id': 'self', 'role': 'candidate', 'text': 'query'}]
        self.assertEqual(rank_vectors(rows, np.ones((5, 2))), {'q': ['a', 'b']})

    def test_zero_vector(self):
        with self.assertRaises(ValueError): rank_vectors([{}], np.zeros((1, 2)))

    def test_invalid_corpus(self):
        with self.assertRaises(ValueError): check_rows([{'id': 'q', 'role': 'query', 'text': ''}])

    def test_publication_gate(self):
        self.assertTrue(inspect('examples/source.HWPX', b'text'))
        self.assertTrue(inspect('examples/source.XLSX', b'text'))
        self.assertTrue(inspect('outside/data.txt', b'text'))
        self.assertTrue(inspect('examples/a.json', b'\x00\xff'))
        self.assertEqual(inspect('examples/rankings.synthetic.json', b'{}'), [])
        for path in ['examples/corpus.json', 'release/pairs.json', 'examples/raw/labels.json',
                     'release/private_evaluation_mapping.json', 'examples/raw/source.txt']:
            self.assertTrue(inspect(path, b'{}'), path)


if __name__ == '__main__': unittest.main()
