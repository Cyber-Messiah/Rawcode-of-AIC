import unittest
from evaluate import box_iou, evaluate


class EvaluationTests(unittest.TestCase):
    def test_geometry(self):
        self.assertEqual(box_iou([0, 0, 1, 1], [0, 0, 1, 1]), 1)
        self.assertEqual(box_iou([0, 0, .5, .5], [.5, .5, 1, 1]), 0)
        self.assertAlmostEqual(box_iou([0, 0, .5, 1], [.25, 0, .75, 1]), 1/3)

    def test_reference_denominator(self):
        refs = {k: {'bbox': [0, 0, 1, 1]} for k in ['ok', 'missing', 'invalid', 'failed']}
        preds = {'ok': refs['ok'], 'invalid': {'bbox': [0, 0, 1000, 1000]},
                 'failed': {'bbox': [0, 0, 1, 1], 'used_fallback': True},
                 'extra': refs['ok']}
        summary, _ = evaluate(preds, refs)
        self.assertEqual(summary['mean_iou'], .25)
        self.assertEqual(summary['accuracy']['IoU>=0.50'], .25)
        self.assertEqual(summary['mean_iou_valid_only'], 1)
        self.assertEqual(summary['ignored_unlabeled_predictions'], 1)
        self.assertEqual(summary['counts'], dict(ok=1, missing=1, invalid=1, failed=1))

    def test_reject_mismatched_questions(self):
        with self.assertRaises(ValueError):
            evaluate({'a': {'bbox': [0, 0, 1, 1], 'query': 'wrong'}},
                     {'a': {'bbox': [0, 0, 1, 1], 'query': 'right'}})


if __name__ == '__main__':
    unittest.main()
