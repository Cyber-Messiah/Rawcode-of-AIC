import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from predict_rgb_all import answer_to_text, build_messages, parse_box, read_progress, run_rows


class RunnerTests(unittest.TestCase):
    def test_model_input_and_output_contract(self):
        image = object()
        messages = build_messages(image, 'red car')
        self.assertIs(messages[0]['content'][0]['image'], image)
        self.assertEqual(messages[0]['content'][1]['text'],
                         'Locate a single instance that matches the following description: red car')
        text = '<box><100><200><300><400></box>'
        for result in (text, [text], (text, [], {'stats': 1}), ([text], [])):
            self.assertEqual(answer_to_text(result, None), text)

    def test_parse(self):
        self.assertEqual(parse_box('<box><100> <200><300><400></box>'), [.1, .2, .3, .4])
        self.assertEqual(parse_box('<box>(100,200,300,400)</box>'), [.1, .2, .3, .4])
        self.assertIsNone(parse_box('<box><100><200><1300><400></box>'))
        self.assertIsNone(parse_box('<box><300><200><100><400></box>'))

    def test_torn_tail_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'predictions.jsonl'
            path.write_bytes(b'{"id":"a","status":"ok"}\n{"id":')
            self.assertEqual(list(read_progress(path)), ['a'])
            with path.open('ab') as f:
                f.write(b'{"id":"b"}\n')
            self.assertEqual(list(read_progress(path)), ['a', 'b'])
            path.write_bytes(b'{"id":"a"}')
            read_progress(path)
            self.assertTrue(path.read_bytes().endswith(b'\n'))
            path.write_bytes(b'broken\n{"id":"b"}\n')
            with self.assertRaises(ValueError):
                read_progress(path)

    def test_9555_rows_failures_restart_and_export(self):
        rows = [(str(i), {'visible': 'image.png', 'query': f'object {i}'}) for i in range(9555)]
        def predict(key, item):
            if key == '42':
                raise RuntimeError('simulated failure')
            return dict(status='ok', bbox=[.1, .2, .3, .4])
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(tmp)
            result = run_rows(rows, {}, output, predict, save_every=1000)
            self.assertEqual(result['successful'], 9554)
            self.assertEqual(result['failed'], 1)
            self.assertFalse(result['complete'])
            called = []
            def retry(key, item):
                called.append(key)
                return dict(status='ok', bbox=[.1, .2, .3, .4])
            result = run_rows(rows, read_progress(output / 'predictions.jsonl'), output, retry)
            self.assertEqual(called, ['42'])
            self.assertTrue(result['complete'])
            self.assertEqual(len(json.loads((output / 'queries_rgb.json').read_text())), 9555)

    def test_keyboard_interrupt_preserves_progress(self):
        rows = [('a', {}), ('b', {})]
        def predict(key, item):
            if key == 'b':
                raise KeyboardInterrupt()
            return dict(status='ok', bbox=[.1, .2, .3, .4])
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            output = Path(tmp)
            with self.assertRaises(KeyboardInterrupt):
                run_rows(rows, {}, output, predict)
            self.assertEqual(list(read_progress(output / 'predictions.jsonl')), ['a'])
            summary = json.loads((output / 'summary.json').read_text())
            self.assertEqual(summary['pending'], 1)


if __name__ == '__main__':
    unittest.main()
