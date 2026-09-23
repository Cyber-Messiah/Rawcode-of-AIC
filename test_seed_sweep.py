import contextlib
import io
import unittest
from seed_sweep import aggregate, run_once


class SweepTests(unittest.TestCase):
    def test_aggregate_and_failure_denominator(self):
        rows = [('a', {}), ('b', {})]
        refs = {key: {'bbox': [0, 0, 1, 1]} for key, _ in rows}
        def predict(key, item):
            if key == 'b':
                raise RuntimeError('failure')
            return dict(status='ok', bbox=[0, 0, 1, 1], answer='not retained')
        with contextlib.redirect_stdout(io.StringIO()):
            run = run_once(rows, refs, predict, 42)
        self.assertEqual(run['mean_iou'], .5)
        self.assertEqual(run['acc_05'], .5)
        self.assertNotIn('predictions', run)
        result = aggregate([run, dict(run, mean_iou=.7)])
        self.assertAlmostEqual(result['mean_iou']['mean'], .6)
        self.assertAlmostEqual(result['mean_iou']['sample_std'], 2**.5 * .1)
        self.assertIsNone(aggregate([run])['mean_iou']['sample_std'])


if __name__ == '__main__':
    unittest.main()
