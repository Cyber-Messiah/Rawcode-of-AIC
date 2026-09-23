import json
from pathlib import Path
import tempfile
import unittest

from prepare_reference_subset import build_subset, load_json, merge_annotations


class ReferenceTests(unittest.TestCase):
    def test_matching_validation_and_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            annotations = root / 'annotations'
            annotations.mkdir()
            image = root / 'Images' / 'visible' / 'a.png'
            image.parent.mkdir(parents=True)
            image.write_bytes(b'copy-test-payload')
            item = dict(visible='Images/visible/a.png', query='object')
            queries = {'a': item, 'b': item, 'unlabeled': item}
            path = annotations / 'one.json'
            answer = dict(item, bbox=[0.1, 0.2, 0.3, 0.4], query_zh='目标')
            path.write_text(json.dumps({'b': answer, 'a': answer}), encoding='utf-8')
            query_path = root / 'queries.json'
            query_path.write_text(json.dumps(queries), encoding='utf-8')
            output = root / 'subset'
            report = build_subset(annotations, query_path, root, output)
            self.assertEqual(report['reference_queries'], 2)
            self.assertEqual(report['total_image_files'], 1)
            subset = load_json(output / 'queries.json')
            self.assertEqual(list(subset), ['a', 'b'])
            self.assertNotIn('bbox', subset['a'])
            self.assertNotIn('query_zh', subset['a'])
            self.assertEqual(load_json(output / 'annotations.json')['a'], answer)
            self.assertEqual((output / item['visible']).read_bytes(), image.read_bytes())
            with self.assertRaises(FileExistsError):
                build_subset(annotations, query_path, root, output)
            for bad in [dict(answer, bbox=[.5, .2, .3, .4]),
                        dict(answer, bbox=[0, 0, 1000, 1000]), dict(answer, query='wrong')]:
                path.write_text(json.dumps({'a': bad}), encoding='utf-8')
                with self.assertRaises(ValueError):
                    merge_annotations(queries, [path])
            path.write_text(json.dumps({'unknown': answer}), encoding='utf-8')
            with self.assertRaises(ValueError):
                merge_annotations(queries, [path])
            path.write_text(json.dumps({'a': answer}), encoding='utf-8')
            with self.assertRaises(ValueError):
                merge_annotations(queries, [path, path])

    def test_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'bad.json'
            path.write_text('{"a": {}, "a": {}}', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_json(path)


if __name__ == '__main__':
    unittest.main()
