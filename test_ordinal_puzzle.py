import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from experiment_f_official_multi import (
    deduplicate, infer_one, make_puzzle, parse_boxes, remove_parent_boxes, select_tile, summarize,
)
from prepare_ordinal_subset import parse_ordinal, select


class OrdinalPuzzleTests(unittest.TestCase):
    def test_selects_clear_queries_without_leaking_answers(self):
        queries = {
            'a': {'visible': 'Images/visible/a.png',
                  'query': 'From left to right, the second stone pier'},
            'b': {'visible': 'Images/visible/b.png',
                  'query': 'The third object from left to right and top to bottom'},
            'c': {'visible': 'Images/visible/c.png', 'query': 'A red sign'},
        }
        references = {key: {**item, 'bbox': [.1, .2, .3, .4]}
                      for key, item in queries.items()}
        subset, answers, parsed, _ = select(queries, references)
        self.assertEqual(list(subset), ['a'])
        self.assertNotIn('bbox', subset['a'])
        self.assertEqual(answers['a']['bbox'], [.1, .2, .3, .4])
        self.assertEqual(parsed['a'], dict(rank=2, direction='left_to_right', target='stone pier'))
        self.assertEqual(parse_ordinal('First planter (from right to left)'),
                         (1, 'right_to_left', 'planter'))
        self.assertEqual(parse_ordinal('The first deer from the left, when counting from left to right.'),
                         (1, 'left_to_right', 'deer'))
        self.assertEqual(parse_ordinal('Count the second tiger from right to left'),
                         (2, 'right_to_left', 'tiger'))

    def test_official_multi_answer_parent_and_duplicate_cleanup(self):
        answer = ('<ref>stone pier</ref><box><100><100><400><400></box>'
                  '<box><100><100><190><190></box>'
                  '<box><300><300><390><390></box>'
                  '<box><101><101><191><191></box>'
                  '<box>none</box>')
        raw = parse_boxes(answer)
        self.assertEqual(len(raw), 4)
        without_parents = remove_parent_boxes(raw)
        self.assertEqual(len(without_parents), 3)
        self.assertEqual(len(deduplicate(without_parents)), 2)

    def test_puzzle_mapping_and_reference_statistics(self):
        boxes = [[.1, .1, .2, .4], [.5, .1, .6, .4]]
        image = Image.new('RGB', (100, 100), 'blue')
        puzzle, spans, used = make_puzzle(image, boxes, tile_height=40, max_width=80)
        self.assertLessEqual(puzzle.width, 80)
        self.assertEqual(used, boxes)
        center = (spans[1][0] + spans[1][1]) / 2
        self.assertEqual(select_tile([center - .01, .1, center + .01, .9], spans), 1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            rows = [('a', {'query': 'From left to right, the second stone pier',
                           'visible': 'Images/visible/a.png'})]
            refs = {'a': {**rows[0][1], 'bbox': boxes[1]}}
            record = {'a': dict(status='ok', bbox=boxes[1], raw_boxes=boxes,
                                after_parent=boxes, after_dedup=boxes)}
            summary = summarize(rows, refs, record, {}, output)
            self.assertEqual(summary['acc_at_05'], 1)
            self.assertEqual(summary['gt_in_raw'], 1)
            self.assertEqual(list(json.loads((output / 'queries_f.json').read_text())), ['a'])

    def test_end_to_end_f_uses_multi_answer_and_maps_puzzle_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / 'Images/visible'
            image_dir.mkdir(parents=True)
            Image.new('RGB', (100, 40), 'blue').save(image_dir / 'a.png')
            item = {'visible': 'Images/visible/a.png',
                    'query': 'From left to right, the second stone pier'}
            reference = {**item, 'bbox': [.5, .1, .6, .4]}
            args = SimpleNamespace(data_root=root, output_dir=root / 'output', seed=42,
                                   temperature=.7, image_token_limit=4096,
                                   multi_max_new_tokens=2048, select_max_new_tokens=128,
                                   parent_containment=.9, parent_min_children=2,
                                   parent_min_area_ratio=1.5, dedup_iou=.5,
                                   tile_height=40, gap=10, max_puzzle_width=200)
            responses = [('<box><100><100><200><400></box>'
                          '<box><500><100><600><400></box>'),
                         '<box><700><100><800><900></box>']
            with patch('experiment_f_official_multi.generate', side_effect=responses) as mocked:
                result = infer_one('a', item, reference, args, None, None, None)
            self.assertEqual(mocked.call_count, 2)
            self.assertIn('Locate all the instances', mocked.call_args_list[0].args[4])
            self.assertEqual(result['status'], 'ok')
            self.assertEqual(result['selected_index'], 1)
            self.assertEqual(result['bbox'], reference['bbox'])
            self.assertTrue((root / 'output/puzzles/a.png').is_file())


if __name__ == '__main__':
    unittest.main()
