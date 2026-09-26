import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

from PIL import Image

import predict_bbox_gprime as gp


def tagged(*boxes):
    return ' '.join('<box><%d><%d><%d><%d></box>' % box for box in boxes)


class GPrimeTests(unittest.TestCase):
    def test_rank_deficit_falls_back_to_original_a_and_single_proposal_is_rejected(self):
        row = dict(id='x', ordinal=True, rank=2, direction='left_to_right',
                   a_bbox=[.7, .2, .8, .3], bbox=[.7, .2, .8, .3],
                   candidate_boxes=[[.1, .2, .2, .3]], source='a_ordinal_fallback')
        new, proposals = gp.stable_new_boxes(
            [[.1, .2, .2, .3]], [(1, [.7, .2, .8, .3])])
        self.assertEqual(new, [])
        self.assertEqual(proposals[0]['support'], 1)
        recalled = dict(accepted_new_boxes=[])
        final = gp.select_final(row, row['candidate_boxes'], [], recalled)
        self.assertEqual(final['bbox'], row['a_bbox'])
        self.assertEqual(final['source'], 'a_ordinal_fallback')

    def test_wide_box_is_replaced_only_by_two_countable_children(self):
        image = Image.new('RGB', (100, 100))
        parent = [.05, .2, .95, .6]
        row = dict(id='wide', rank=2, target='sign', candidate_boxes=[parent])
        args = types.SimpleNamespace(seed=42, multi_max_new_tokens=128,
                                     multi_image_token_limit=4096,
                                     gprime_temperature=.4)
        calls = []

        def generate(image, prompt, seed, max_tokens, limit, temperature):
            calls.append((prompt, temperature))
            return tagged((100, 200, 200, 800), (600, 200, 700, 800))

        boxes, steps = gp.refine_wide(row, image, generate, args)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][1], .4)
        self.assertEqual(len(boxes), 2)
        self.assertTrue(steps[0]['accepted'])
        self.assertFalse(any(gp.base.iou(parent, box) >= .999 for box in boxes))

    def test_full_pipeline_recall_and_submission_format(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            data = root / 'data'
            image_path = data / 'Images' / 'visible' / 'sample.png'
            image_path.parent.mkdir(parents=True)
            Image.new('RGB', (100, 100), 'white').save(image_path)
            relative = 'Images/visible/sample.png'
            queries = {
                'plain': dict(visible=relative, query='a red sign'),
                'second': dict(visible=relative,
                               query='the second sign from left to right'),
                'third': dict(visible=relative,
                              query='the third sign from left to right'),
            }
            (data / 'queries.json').write_text(json.dumps(queries), encoding='utf-8')
            output = root / 'out'
            calls = []

            def generate(image, prompt, seed, max_tokens, limit, temperature=None):
                calls.append((prompt, temperature))
                if prompt.startswith('Locate a single'):
                    return tagged((100, 200, 200, 300))
                if prompt.startswith('Locate all the instances'):
                    return tagged((100, 200, 200, 300), (400, 200, 500, 300))
                return tagged((700, 200, 800, 300))

            argv = ['--data-root', str(data), '--model-path', str(root / 'model'),
                    '--output-dir', str(output), '--save-every', '1']
            with patch.object(gp.base, 'make_generator', return_value=generate):
                self.assertEqual(gp.main(argv), 0)
                before = len(calls)
                self.assertEqual(gp.main(argv), 0)
                self.assertEqual(len(calls), before)

            submission = json.loads((output / 'submission_complete.json').read_text())
            self.assertEqual(set(submission), set(queries))
            self.assertEqual(submission['plain']['bbox'], [.1, .2, .2, .3])
            self.assertEqual(submission['second']['bbox'], [.4, .2, .5, .3])
            self.assertEqual(submission['third']['bbox'], [.7, .2, .8, .3])
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['total_queries'], 3)
            self.assertEqual(summary['ordinal_queries'], 2)
            self.assertTrue(summary['submission_ready'])
            self.assertEqual(summary['recall_triggered_groups'], 1)
            self.assertEqual(summary['sources']['gprime_recall'], 2)
            self.assertEqual(len([call for call in calls if call[1] == .4]), 2)


if __name__ == '__main__':
    unittest.main()
