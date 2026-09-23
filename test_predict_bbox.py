import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from predict_bbox import export_results, find_queries, parse_box, read_progress, run_rows, validate_inputs


class PredictionTests(unittest.TestCase):
    def test_competition_layout_and_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'queries').mkdir()
            (root / 'Images/visible').mkdir(parents=True)
            (root / 'Images/visible/a.png').write_bytes(b'image')
            items = {'a_1': {'visible': 'Images/visible/a.png', 'infrared': 'Images/infrared/a.png',
                             'depth': 'Images/depth/a.png', 'query': 'the red object'},
                     'a_2': {'visible': 'Images/visible/a.png', 'infrared': 'Images/infrared/a.png',
                             'depth': 'Images/depth/a.png', 'query': 'the blue object'}}
            (root / 'queries/queries.json').write_text(json.dumps(items), encoding='utf-8')
            rows = validate_inputs(find_queries(root, None), root)
            self.assertEqual(len(rows), 2)
            output = root / 'output'
            output.mkdir()
            def predict(index, key, item):
                if index == 1:
                    return dict(status='error', bbox=None, errors=['no box'])
                return dict(status='ok', bbox=[.1, .2, .3, .4])
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run_rows(rows, {}, output, predict, False, 1)
            self.assertFalse(summary['submission_ready'])
            self.assertFalse((output / 'submission_complete.json').exists())
            self.assertEqual(list(read_progress(output / 'predictions.jsonl')), ['a_1', 'a_2'])
            result = export_results(rows, read_progress(output / 'predictions.jsonl'), output, True)
            self.assertEqual(result['fallback_boxes'], 1)
            submission = json.loads((output / 'submission_complete.json').read_text(encoding='utf-8'))
            self.assertEqual(submission['a_1']['bbox'], [.1, .2, .3, .4])
            self.assertEqual(submission['a_2']['bbox'], [0, 0, 1, 1])
            self.assertEqual(submission['a_1']['infrared'], items['a_1']['infrared'])
            self.assertEqual(list(submission), list(items))

    def test_parser_and_input_validation(self):
        self.assertEqual(parse_box('<box><100><200><300><400></box>'), [.1, .2, .3, .4])
        self.assertEqual(parse_box('<box>(100,200,300,400)</box>'), [.1, .2, .3, .4])
        self.assertIsNone(parse_box('<box><300><200><100><400></box>'))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'queries.json').write_text(json.dumps({'a': {'query': 'x', 'visible': '../outside.png'}}))
            with self.assertRaises(ValueError):
                validate_inputs(find_queries(root, None), root)
            (root / 'queries').mkdir()
            (root / 'queries/queries.json').write_text('{}')
            with self.assertRaises(ValueError):
                find_queries(root, None)

    def test_images_symlink_outside_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            parent = Path(tmp)
            root = parent / 'dataset'
            image_dir = parent / 'images' / 'visible'
            root.mkdir()
            image_dir.mkdir(parents=True)
            (image_dir / 'a.png').write_bytes(b'image')
            try:
                (root / 'Images').symlink_to(image_dir.parent, target_is_directory=True)
            except OSError as exc:
                self.skipTest(f'Symlinks unavailable: {exc}')
            (root / 'queries.json').write_text(json.dumps({
                'a': {'query': 'x', 'visible': 'Images/visible/a.png'}
            }), encoding='utf-8')
            self.assertEqual(len(validate_inputs(root / 'queries.json', root)), 1)

    def test_torn_progress_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictions.jsonl'
            path.write_bytes(b'{"id":"a","status":"ok"}\n{"id":')
            self.assertEqual(list(read_progress(path)), ['a'])
            self.assertEqual(path.read_bytes(), b'{"id":"a","status":"ok"}\n')


if __name__ == '__main__':
    unittest.main()
