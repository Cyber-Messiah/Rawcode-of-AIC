#!/usr/bin/env python3
"""Merge human boxes and build a self-contained, answer-free query subset."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile


def load_json(path):
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f'{path}: duplicate JSON key {key}')
            result[key] = value
        return result
    data = json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique_pairs)
    if not isinstance(data, dict):
        raise ValueError(f'{path}: expected an object keyed by sample ID')
    return data


def valid_box(box):
    return (isinstance(box, list) and len(box) == 4
            and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in box)
            and box[0] < box[2] and box[1] < box[3])


def merge_annotations(queries, paths):
    answers, provenance, counts = {}, {}, {}
    for path in paths:
        annotations = load_json(path)
        counts[path.name] = len(annotations)
        for key, annotation in annotations.items():
            if key not in queries:
                raise ValueError(f'{path.name}: unknown query ID {key}')
            if not isinstance(annotation, dict) or not valid_box(annotation.get('bbox')):
                raise ValueError(f'{path.name}: invalid normalized xyxy bbox for {key}')
            source = queries[key]
            for field in ('visible', 'infrared', 'depth', 'query'):
                if annotation.get(field) != source.get(field):
                    raise ValueError(f'{path.name}: {key} differs from original {field}')
            if key in answers:
                raise ValueError(f'{key}: repeated annotation in {provenance[key]} and {path.name}; resolve explicitly')
            answers[key] = {**source, **annotation}
            provenance[key] = path.name
    # Keep original question order, IDs, text and paths. Never put answers in input queries.
    subset = {key: {k: v for k, v in item.items() if k != 'bbox'}
              for key, item in queries.items() if key in answers}
    answers = {key: answers[key] for key in subset}
    return answers, subset, provenance, counts


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def build_subset(annotation_dir, queries_path, data_root, output):
    paths = sorted(annotation_dir.glob('*.json'))
    if not paths:
        raise ValueError('No annotation JSON files found')
    if output.exists():
        raise FileExistsError(f'Output already exists; choose a new directory: {output}')
    queries = load_json(queries_path)
    answers, subset, provenance, counts = merge_annotations(queries, paths)
    if not answers:
        raise ValueError('No reference answers found')
    images = {}
    by_modality = {}
    data_root = data_root.resolve()
    for field in ('visible', 'infrared', 'depth'):
        relative_paths = sorted({item[field] for item in subset.values() if item.get(field)})
        by_modality[field] = len(relative_paths)
        for relative in relative_paths:
            rel = Path(relative)
            source = (data_root / rel).resolve()
            if rel.is_absolute() or '..' in rel.parts or not source.is_relative_to(data_root):
                raise ValueError(f'Unsafe image path: {relative}')
            if not source.is_file():
                raise FileNotFoundError(source)
            images[relative] = source
    report = dict(original_queries=len(queries), reference_queries=len(answers),
                  excluded_queries=len(queries) - len(answers), annotations_per_file=counts,
                  images_per_modality=by_modality, total_image_files=len(images),
                  bbox_format='normalized xyxy [x1, y1, x2, y2], range 0..1',
                  source_sha256={str(p.resolve()): sha256(p) for p in [queries_path, *paths]},
                  annotation_sources=provenance)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after every image has been copied and verified.
    with tempfile.TemporaryDirectory(prefix='reference-build-', dir=output.parent) as tmp:
        staging = Path(tmp) / 'dataset'
        staging.mkdir()
        write_json(staging / 'annotations.json', answers)
        write_json(staging / 'queries.json', subset)
        hashes = {}
        for number, (relative, source) in enumerate(sorted(images.items()), 1):
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            digest = sha256(source)
            if sha256(target) != digest:
                raise IOError(f'Copy verification failed: {relative}')
            hashes[relative] = digest
            if number % 200 == 0:
                print(f'Copied and verified {number}/{len(images)} images', flush=True)
        report['image_sha256'] = hashes
        write_json(staging / 'manifest.json', report)
        (staging / 'README.md').write_text(
            '# 人工参考答案子集\n\n'
            f'包含 {len(answers)} 道题，{len(images)} 个图片文件。\n\n'
            '- annotations.json：人工参考答案总表，保留 bbox 和原有中文翻译。\n'
            '- queries.json：同一组题目，不含 bbox，按原题顺序排列。\n'
            '- Images/：题目引用的图片，保留原始相对路径及全部模态。\n'
            '- manifest.json：来源、统计及每个图片的 SHA-256 校验值。\n\n'
            'bbox 为归一化 [x1,y1,x2,y2]，未缩放、裁剪或重新量化。\n'
            '预测时将 --annotation 指向 queries.json，--data-root 指向本目录，'
            '并使用独立 --output-dir。评测按题目 ID 对齐 annotations.json。\n',
            encoding='utf-8')
        staging.rename(output)
    return {k: v for k, v in report.items() if k not in ('image_sha256', 'annotation_sources', 'source_sha256')}


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--annotations-dir', type=Path, default=root / 'row_check_json')
    parser.add_argument('--queries', type=Path, default=root / 'datasets' / 'full' / 'queries.json')
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--output-dir', type=Path, default=root / 'datasets' / 'reference_subset')
    args = parser.parse_args()
    if args.queries is None:
        parser.error('Specify --queries')
    summary = build_subset(args.annotations_dir, args.queries,
                           args.data_root or args.queries.parent, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
