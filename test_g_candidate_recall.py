import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from experiment_g_candidate_recall import (
    apply_recall, candidate_gate, group_records, recall_group, same_instance, search_regions,
    stable_new_boxes,
)


def args(root):
    return SimpleNamespace(data_root=root, seed=42, temperature=.4,
                           temperature_delta=.15, brightness_delta=.08,
                           image_token_limit=4096, multi_max_new_tokens=2048,
                           max_calls_per_group=12, max_gap_regions=4,
                           min_support=2, min_size_ratio=.15,
                           max_size_ratio=6, wide_width_threshold=.8,
                           large_area_threshold=.1, probe_complete=False)


class CandidateRecallTests(unittest.TestCase):
    def test_same_image_target_uses_max_rank_as_lower_bound(self):
        rows = [dict(id='one', visible='a.png', target='trough', rank=1,
                     refined_boxes=[[.4, .2, .55, .3]]),
                dict(id='two', visible='a.png', target='trough', rank=2,
                     refined_boxes=[[.4, .2, .55, .3]])]
        groups = group_records({row['id']: row for row in rows})
        seed = rows[0]['refined_boxes']
        self.assertEqual(len(groups), 1)
        self.assertEqual(candidate_gate(rows, seed), (2, 'rank_deficit'))

    def test_similar_box_and_neighbor_remain_distinct(self):
        box = [.393, .378, .410, .420]
        self.assertTrue(same_instance(box, [.394, .379, .411, .421]))
        self.assertFalse(same_instance(box, [.415, .356, .445, .408]))

    def test_one_off_candidate_does_not_enter_g(self):
        seed = [[.4, .2, .55, .3]]
        other = [.03, .17, .21, .26]
        found, proposals = stable_new_boxes(seed, [(1, other)], 2, .15, 6)
        self.assertEqual(found, [])
        self.assertEqual(proposals[0]['support'], 1)
        found, proposals = stable_new_boxes(
            seed, [(1, other), (2, [.031, .171, .211, .261])], 2, .15, 6)
        self.assertEqual(len(found), 1)
        self.assertEqual(proposals[0]['support'], 2)

    def test_failed_recall_preserves_each_query_candidates(self):
        original = dict(id='one', refined_boxes=[[.4, .2, .55, .3]])
        recalled = dict(initial_boxes=original['refined_boxes'],
                        final_boxes=[[.4, .2, .55, .3], [.03, .17, .21, .26]],
                        accepted_new_boxes=[])
        updated = apply_recall(original, recalled)
        self.assertEqual(updated['refined_boxes'], original['refined_boxes'])

    def test_found_instance_is_excluded_from_complementary_regions(self):
        regions = search_regions(Image.new('RGB', (1000, 500)),
                                 [[.4, .2, .55, .3]])
        gaps = [bounds for name, bounds in regions if name.startswith('uncovered')]
        self.assertEqual(len(gaps), 2)
        self.assertTrue(any(bounds[2] < 410 for bounds in gaps))
        self.assertTrue(any(bounds[0] > 540 for bounds in gaps))

    def test_two_independent_answers_recover_left_trough(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'Images/visible/sample.png'
            path.parent.mkdir(parents=True)
            Image.new('RGB', (1000, 500), 'white').save(path)
            rows = [dict(id='first', visible='Images/visible/sample.png',
                         target='trough', rank=1,
                         refined_boxes=[[.4, .2, .55, .3]]),
                    dict(id='second', visible='Images/visible/sample.png',
                         target='trough', rank=2,
                         refined_boxes=[[.4, .2, .55, .3]])]
            answer = '<box><30><170><210><260></box>'
            with patch('experiment_g_candidate_recall.generate',
                       side_effect=[answer, answer]) as mock:
                result = recall_group(rows, args(root), None, None, None)
            self.assertEqual(mock.call_count, 2)
            self.assertEqual(len(result['final_boxes']), 2)
            self.assertEqual(len(result['attempts']), 2)
            self.assertLess(result['final_boxes'][1][0], .1)


if __name__ == '__main__':
    unittest.main()
