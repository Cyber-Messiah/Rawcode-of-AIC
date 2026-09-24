import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from types import SimpleNamespace

import predict_bbox_v2 as v2


class CoordinateTests(unittest.TestCase):
    def test_tagged_model_units_and_reversed_corners(self):
        boxes, audit = v2.parse_boxes('<box><800><700><200><100></box>')
        self.assertEqual(boxes, [[0.2, 0.1, 0.8, 0.7]])
        self.assertEqual(audit['reversed_corners'], 1)
        self.assertEqual(audit['units'], {'model_1000': 1})

    def test_normalized_plain_box_is_not_divided_twice(self):
        boxes, audit = v2.parse_boxes('<box>(0.8, 0.7, 0.2, 0.1)</box>')
        self.assertEqual(boxes, [[0.2, 0.1, 0.8, 0.7]])
        self.assertEqual(audit['units'], {'normalized': 1})

    def test_pixel_box_and_tagged_box(self):
        boxes, audit = v2.parse_boxes('<box>[1600, 900, 400, 100]</box>'
                                      '<box><100><200><300><400></box>', (2000, 1000))
        self.assertEqual(boxes, [[0.2, 0.1, 0.8, 0.9], [0.1, 0.2, 0.3, 0.4]])
        self.assertEqual(audit['units'], {'pixel': 1, 'model_1000': 1})

    def test_invalid_and_none_are_audited(self):
        boxes, audit = v2.parse_boxes('<box><500><500><500><600></box>'
                                      '<box><1200><0><500><500></box><box>None</box>')
        self.assertEqual(boxes, [])
        self.assertEqual(audit['rejected'], {'degenerate_or_out_of_range': 1,
                                             'tagged_out_of_range': 1})
        self.assertEqual(audit['explicit_none'], 1)

    def test_g_respects_direction(self):
        boxes = [[0.7, 0.1, 0.8, 0.2], [0.1, 0.1, 0.2, 0.2],
                 [0.4, 0.1, 0.5, 0.2]]
        self.assertEqual(v2.select_g(boxes, 2, 'left_to_right'), boxes[2])
        self.assertEqual(v2.select_g(boxes, 1, 'right_to_left'), boxes[0])
        self.assertIsNone(v2.select_g(boxes, 4, 'left_to_right'))

    def test_crop_mapping_uses_original_dimensions(self):
        self.assertEqual(v2.map_from_crop([0.2, 0.1, 0.8, 0.9],
                                          (100, 50, 300, 150), (1000, 500)),
                         [0.14, 0.12, 0.26, 0.28])

    def test_ordinal_inference_chooses_g_and_nonordinal_chooses_a(self):
        from PIL import Image
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new('RGB', (200, 100), 'white').save(root / 'image.png')
            args = SimpleNamespace(data_root=root, retries=0, seed=42,
                                   max_new_tokens=128, image_token_limit=4096,
                                   multi_max_new_tokens=2048,
                                   multi_image_token_limit=4096, max_depth=0)
            a = '<box><900><800><700><600></box>'
            multi = ('<box><100><100><200><200></box>'
                     '<box><400><100><500><200></box>'
                     '<box><700><100><800><200></box>')
            calls = []
            def generate(image, prompt, seed, max_tokens, token_limit):
                calls.append(prompt)
                return multi if prompt.startswith('Locate all') else a
            ordinal = v2.infer(0, 'one', {'visible': 'image.png',
                'query': 'second window from right to left'}, args, generate)
            self.assertEqual(ordinal['bbox'], [0.4, 0.1, 0.5, 0.2])
            self.assertEqual(ordinal['source'], 'g_initial')
            self.assertEqual(ordinal['a_audit']['reversed_corners'], 1)
            nonordinal = v2.infer(1, 'two', {'visible': 'image.png',
                'query': 'a window'}, args, generate)
            self.assertEqual(nonordinal['bbox'], [0.7, 0.6, 0.9, 0.8])
            self.assertEqual(nonordinal['source'], 'a')
            self.assertEqual(len(calls), 3)

    def test_submission_contains_canonical_boxes_only(self):
        import json
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            rows = [('one', {'visible': 'image.png', 'query': 'a window'})]
            records = {'one': {'status': 'ok', 'bbox': [0.2, 0.1, 0.8, 0.7],
                               'source': 'a', 'a_audit': {'reversed_corners': 1}}}
            summary = v2.export(rows, records, output, False)
            self.assertTrue(summary['submission_ready'])
            self.assertEqual(summary['reversed_corner_boxes'], 1)
            submission = json.loads((output / 'submission_complete.json').read_text())
            self.assertEqual(submission['one']['bbox'], [0.2, 0.1, 0.8, 0.7])


if __name__ == '__main__':
    unittest.main()
