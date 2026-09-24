import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from experiment_f_recursive_multi import (
    crop_region, effective_children, infer_one, map_from_crop, replace_parent,
    summarize, suspicious_parent,
)


def arguments(root):
    return SimpleNamespace(data_root=root, output_dir=root / 'out', seed=42,
                           temperature=.7, image_token_limit=4096,
                           multi_max_new_tokens=2048, select_max_new_tokens=128,
                           parent_containment=.9, parent_min_children=2,
                           parent_min_area_ratio=1.5, dedup_iou=.5,
                           tile_height=40, gap=10, max_puzzle_width=200,
                           max_depth=1, single_area_threshold=.10,
                           multi_area_threshold=.10, relative_area_ratio=1.75,
                           a_area_ratio=3, a_containment=.9,
                           a_min_area_threshold=.02,
                           recursive_global_area_threshold=.10,
                           child_max_area_ratio=.8, child_min_containment=.9,
                           crop_padding=0)


class RecursiveOrdinalTests(unittest.TestCase):
    def test_calibrated_outlier_and_a_scale_triggers(self):
        self.assertEqual(suspicious_parent([[.1, .1, .5, .5]])[0], 0)
        self.assertIsNone(suspicious_parent([[.1, .1, .2, .2]])[0])
        boxes = [[0, 0, .2, .2], [.3, 0, .4, .2]]
        self.assertIsNone(suspicious_parent(boxes)[0])
        self.assertEqual(suspicious_parent([[0, 0, .4, .4], [.5, 0, .6, .2]]),
                         (0, ['relative_area_outlier']))
        self.assertEqual(suspicious_parent([[0, 0, .15, .15], [.3, 0, .4, .1]],
                                           baseline_box=[.01, .01, .05, .05])[1],
                         ['encloses_smaller_a_box'])
        self.assertIsNone(suspicious_parent([[0, 0, .1, .1]],
                                             baseline_box=[.01, .01, .03, .03])[0])

    def test_crop_mapping_and_repeated_whole_crop_rejection(self):
        image = Image.new('RGB', (100, 100), 'blue')
        parent = [.2, .2, .8, .8]
        crop, bounds = crop_region(image, parent, padding=0)
        self.assertEqual(crop.size, (60, 60))
        self.assertEqual(bounds, (20, 20, 80, 80))
        mapped = map_from_crop([.25, .25, .5, .5], bounds, image.size)
        self.assertAlmostEqual(mapped[0], .35)
        self.assertAlmostEqual(mapped[2], .5)
        children = effective_children([[0, 0, 1, 1], [.25, .25, .5, .5]],
                                      bounds, image.size, parent)
        self.assertEqual(len(children), 1)
        self.assertEqual(replace_parent([parent, [.85, .2, .95, .8]], parent,
                                        [children[0][1]], .5)[0], children[0][1])

    def test_none_uses_a_without_selection_or_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Images/visible').mkdir(parents=True)
            Image.new('RGB', (100, 100), 'white').save(root / 'Images/visible/a.png')
            item = dict(query='The first person from left to right',
                        visible='Images/visible/a.png')
            a_box = [.1, .1, .2, .3]
            with patch('experiment_f_recursive_multi.generate', return_value='<box>None</box>') as mocked:
                record = infer_one('a', item, a_box, arguments(root), None, None, None)
            self.assertEqual(mocked.call_count, 1)
            self.assertEqual(record['status'], 'fallback_a_none')
            self.assertEqual(record['bbox'], a_box)

    def test_refines_parent_then_selects_mapped_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Images/visible').mkdir(parents=True)
            Image.new('RGB', (100, 100), 'white').save(root / 'Images/visible/a.png')
            item = dict(query='The second person from left to right',
                        visible='Images/visible/a.png')
            args = arguments(root)
            answers = [
                '<box><0><0><1000><1000></box>',
                '<box><0><0><1000><1000></box>',
                '<box><100><100><250><300></box><box><600><100><750><300></box>',
                '<box><700><0><800><1000></box>',
            ]
            with patch('experiment_f_recursive_multi.generate', side_effect=answers) as mocked:
                record = infer_one('a', item, [.6, .1, .75, .3], args, None, None, None)
            self.assertEqual(mocked.call_count, 4)
            self.assertEqual(record['final_source'], 'refined_f')
            self.assertEqual(record['bbox'], [.6, .1, .75, .3])
            self.assertEqual(len(record['refinement_steps']), 1)
            self.assertEqual(len(record['refined_boxes']), 2)
            self.assertEqual(record['original_boxes'], [[0, 0, 1, 1]])
            refs = {'a': {**item, 'bbox': [.6, .1, .75, .3]}}
            summary = summarize([('a', item)], refs, {'a': record},
                                {'a': {'bbox': [.1, .1, .2, .3]}}, args.output_dir)
            self.assertEqual(summary['final_acc_at_05'], 1)
            self.assertEqual(summary['refinement_triggered'], 1)
            self.assertEqual(list(json.loads((args.output_dir / 'queries_recursive.json').read_text())), ['a'])

    def test_stops_after_three_meaningful_shrinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Images/visible').mkdir(parents=True)
            Image.new('RGB', (100, 100), 'white').save(root / 'Images/visible/a.png')
            item = dict(query='The first person from left to right',
                        visible='Images/visible/a.png')
            args = arguments(root)
            args.max_depth = 3
            answers = ['<box><0><0><1000><1000></box>',
                       '<box><0><0><1000><1000></box>',
                       *(['<box><100><100><800><800></box>'] * 3),
                       '<box><0><0><1000><1000></box>']
            with patch('experiment_f_recursive_multi.generate', side_effect=answers) as mocked:
                record = infer_one('a', item, [.1, .1, .2, .2], args, None, None, None)
            self.assertEqual(mocked.call_count, 6)
            self.assertEqual(len(record['refinement_steps']), 3)
            self.assertEqual(record['status'], 'ok')
            self.assertLess(record['bbox'][2] - record['bbox'][0], .35)

    def test_crop_does_not_retrigger_tiny_original_image_box(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'Images/visible').mkdir(parents=True)
            Image.new('RGB', (100, 100), 'white').save(root / 'Images/visible/a.png')
            item = dict(query='The first person from left to right',
                        visible='Images/visible/a.png')
            args = arguments(root)
            args.max_depth = 3
            answers = ['<box><0><0><200><200></box>',
                       '<box><0><0><1000><1000></box>',
                       '<box><0><0><500><500></box>',
                       '<box><0><0><1000><1000></box>']
            with patch('experiment_f_recursive_multi.generate', side_effect=answers) as mocked:
                record = infer_one('a', item, [.01, .01, .03, .03], args, None, None, None)
            self.assertEqual(mocked.call_count, 4)
            self.assertEqual(len(record['refinement_steps']), 1)


if __name__ == '__main__':
    unittest.main()
