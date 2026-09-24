import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from experiment_g_wide_refine import (
    enough_children, horizontal_tiles, refine_record, wide_indices,
)


def args(root):
    return SimpleNamespace(data_root=root, seed=42, temperature=.4,
                           image_token_limit=4096, multi_max_new_tokens=2048,
                           crop_padding=0, wide_width_threshold=.8,
                           wide_child_max_width_ratio=.6,
                           wide_child_max_area_ratio=.45,
                           wide_child_min_containment=.9,
                           wide_dedup_iou=.4, wide_tile_fraction=.42,
                           wide_tile_overlap=.15, wide_tile_max_local_width=.85)


def record():
    parent = [0, .2, 1, .4]
    return dict(id='sample', visible='Images/visible/sample.png', rank=2,
                direction='left_to_right', target='person', refined_boxes=[parent],
                bbox=[.1, .1, .2, .2], final_source='a_puzzle')


class WideRefineTests(unittest.TestCase):
    def test_width_gate_catches_thin_full_row(self):
        boxes = [[0, .2, .81, .23], [.1, .5, .3, .7]]
        self.assertEqual(wide_indices(boxes, .8), [0])

    def test_strict_second_prompt_splits_group_without_changing_f_answer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'Images/visible/sample.png'
            path.parent.mkdir(parents=True)
            Image.new('RGB', (100, 100), 'white').save(path)
            response = ('<box><50><100><250><900></box>'
                        '<box><700><100><900><900></box>')
            with patch('experiment_g_wide_refine.generate', side_effect=[
                    '<box><0><0><1000><1000></box>', response]) as mocked:
                result = refine_record(record(), args(root), None, None, None)
            self.assertEqual(mocked.call_count, 2)
            self.assertEqual(len(result['refined_boxes']), 2)
            self.assertTrue(result['wide_refinement_steps'][0]['accepted'])
            self.assertEqual(result['bbox'], record()['bbox'])
            self.assertEqual(result['final_source'], 'a_puzzle')

    def test_tiled_retry_keeps_parent_if_every_answer_is_whole_crop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'Images/visible/sample.png'
            path.parent.mkdir(parents=True)
            Image.new('RGB', (100, 100), 'white').save(path)
            with patch('experiment_g_wide_refine.generate',
                       return_value='<box><0><0><1000><1000></box>') as mocked:
                result = refine_record(record(), args(root), None, None, None)
            self.assertGreater(mocked.call_count, 2)
            self.assertEqual(result['refined_boxes'], record()['refined_boxes'])
            self.assertFalse(result['wide_refinement_steps'][0]['accepted'])
            self.assertTrue(any(a['kind'].startswith('horizontal_tile_')
                                for a in result['wide_refinement_steps'][0]['attempts']))

    def test_horizontal_tiles_can_supply_distinct_children(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'Images/visible/sample.png'
            path.parent.mkdir(parents=True)
            Image.new('RGB', (100, 100), 'white').save(path)
            whole = '<box><0><0><1000><1000></box>'
            single = '<box><100><100><300><900></box>'
            with patch('experiment_g_wide_refine.generate', side_effect=[
                    whole, whole, single, single]) as mocked:
                result = refine_record(record(), args(root), None, None, None)
            self.assertEqual(mocked.call_count, 4)
            self.assertEqual(len(result['refined_boxes']), 2)
            self.assertLess(result['refined_boxes'][0][2], result['refined_boxes'][1][0])
            self.assertEqual(result['wide_refinement_steps'][0]['attempts'][-1]['kind'],
                             'horizontal_tile_2')

    def test_partial_split_must_be_countable_for_rank(self):
        parent = [0, .2, 1, .4]
        children = [[.05, .22, .2, .38], [.7, .22, .9, .38]]
        self.assertFalse(enough_children([parent], parent, children, 3, .4))
        self.assertTrue(enough_children([parent], parent, children, 2, .4))

    def test_horizontal_tiles_cover_parent_endpoints(self):
        image = Image.new('RGB', (100, 100), 'white')
        tiles = horizontal_tiles(image, [0, .2, 1, .4], padding=0)
        self.assertEqual(tiles[0][1][0], 0)
        self.assertEqual(tiles[-1][1][2], 100)
        self.assertGreaterEqual(len(tiles), 3)


if __name__ == '__main__':
    unittest.main()
