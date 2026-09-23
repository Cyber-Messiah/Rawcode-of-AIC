#!/usr/bin/env python3
"""Evaluate a puzzle strategy using LocateAnything's official multi-instance prompt.

Only explicit horizontal ordinal queries are accepted. Ground truth is used for
evaluation after inference, never to generate or filter candidates.
"""
import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import statistics

from evaluate import box_iou
from predict_rgb_all import answer_to_text, atomic_json, read_progress
from prepare_ordinal_subset import ORDINAL_VALUES, parse_ordinal
from prepare_reference_subset import load_json, sha256, valid_box


BOX_TAG = re.compile(r'<box>(.*?)</box>', re.S)
TAGGED_COORDS = re.compile(r'\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*')
PLAIN_COORDS = re.compile(r'\s*\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?\s*')
ORDINAL_WORDS = {number: word for word, number in ORDINAL_VALUES.items()}


def parse_boxes(answer):
    """Read every four-coordinate box, preserving the model's output order."""
    boxes = []
    for match in BOX_TAG.finditer(answer):
        found = TAGGED_COORDS.fullmatch(match.group(1)) or PLAIN_COORDS.fullmatch(match.group(1))
        if found:
            box = [int(value) / 1000 for value in found.groups()]
            if valid_box(box):
                boxes.append(box)
    return boxes


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def remove_parent_boxes(boxes, containment=0.9, min_children=2, min_area_ratio=1.5,
                        child_duplicate_iou=0.5):
    """Discard a large box containing distinct smaller candidate instances."""
    kept = []
    for index, outer in enumerate(boxes):
        children = []
        for other_index, inner in enumerate(boxes):
            if index == other_index or area(outer) < min_area_ratio * area(inner):
                continue
            if intersection(outer, inner) / area(inner) < containment:
                continue
            if all(box_iou(inner, previous) <= child_duplicate_iou for previous in children):
                children.append(inner)
        if len(children) < min_children:
            kept.append(outer)
    return kept


def deduplicate(boxes, threshold=0.5):
    kept = []
    for box in boxes:
        if all(box_iou(box, previous) <= threshold for previous in kept):
            kept.append(box)
    return kept


def make_puzzle(image, boxes, tile_height=224, gap=12, max_width=1536):
    """Return puzzle, normalized tile spans, and the corresponding valid boxes."""
    from PIL import Image, ImageDraw

    width, height = image.size
    tiles, used_boxes = [], []
    for box in boxes:
        left = max(0, min(width - 1, math.floor(box[0] * width)))
        top = max(0, min(height - 1, math.floor(box[1] * height)))
        right = max(left + 1, min(width, math.ceil(box[2] * width)))
        bottom = max(top + 1, min(height, math.ceil(box[3] * height)))
        crop = image.crop((left, top, right, bottom))
        tile_width = max(1, round(crop.width * tile_height / crop.height))
        if tile_width > max_width:
            fitted_height = max(1, round(crop.height * max_width / crop.width))
            fitted = crop.resize((max_width, fitted_height), Image.Resampling.BICUBIC)
            tile = Image.new('RGB', (max_width, tile_height), 'white')
            tile.paste(fitted, (0, (tile_height - fitted_height) // 2))
        else:
            tile = crop.resize((tile_width, tile_height), Image.Resampling.BICUBIC)
        tiles.append(tile)
        used_boxes.append(box)
    if not tiles:
        return None, [], []
    total_width = sum(tile.width for tile in tiles) + gap * (len(tiles) - 1)
    puzzle = Image.new('RGB', (total_width, tile_height), 'white')
    spans = []
    x = 0
    draw = ImageDraw.Draw(puzzle)
    for tile in tiles:
        puzzle.paste(tile, (x, 0))
        spans.append((x / total_width, (x + tile.width) / total_width))
        draw.rectangle((x, 0, x + tile.width - 1, tile_height - 1), outline='yellow', width=3)
        x += tile.width + gap
    if total_width > max_width:
        new_height = max(1, round(tile_height * max_width / total_width))
        puzzle = puzzle.resize((max_width, new_height), Image.Resampling.BICUBIC)
    return puzzle, spans, used_boxes


def select_tile(box, spans):
    """Map a puzzle box to a tile; a response in a separator picks the nearest tile."""
    if not valid_box(box) or not spans:
        return None
    center = (box[0] + box[2]) / 2
    return min(range(len(spans)), key=lambda i: (
        max(spans[i][0] - center, 0, center - spans[i][1]),
        abs(center - (spans[i][0] + spans[i][1]) / 2)))


def gt_count(boxes, reference, threshold=0.5):
    return sum(box_iou(box, reference) >= threshold for box in boxes)


def draw_puzzle_result(puzzle, spans, candidates, reference, selected_index, path):
    from PIL import ImageDraw

    marked = puzzle.copy()
    draw = ImageDraw.Draw(marked)
    width, height = marked.size
    for index, (left, right) in enumerate(spans):
        x1, x2 = round(left * width), round(right * width) - 1
        if box_iou(candidates[index], reference) >= 0.5:
            draw.rectangle((x1, 0, x2, height - 1), outline='lime', width=5)
        if index == selected_index and x2 - x1 >= 12 and height >= 12:
            draw.rectangle((x1 + 5, 5, x2 - 5, height - 6), outline='red', width=4)
    path.parent.mkdir(parents=True, exist_ok=True)
    marked.save(path)


def load_model(model_path, image_token_limit):
    import torch
    from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; use --check-only for CPU validation')
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config._attn_implementation = 'sdpa'
    config.text_config._attn_implementation = 'sdpa'
    config.text_config._attn_implementation_internal = 'sdpa'
    config.vision_config._attn_implementation = 'sdpa'
    model = AutoModel.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16,
                                     trust_remote_code=True)
    model.language_model.model._attn_implementation = 'sdpa'
    model = model.to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer = tokenizer
    processor.image_processor.in_token_limit = image_token_limit
    return model, tokenizer, processor


def generate(model, tokenizer, processor, image, prompt, max_new_tokens, seed, temperature,
             image_token_limit):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    processor.image_processor.in_token_limit = image_token_limit
    messages = [{'role': 'user', 'content': [
        {'type': 'image', 'image': image}, {'type': 'text', 'text': prompt},
    ]}]
    text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = processor.process_vision_info(messages)
    inputs = processor(text=[text], images=images, videos=videos, return_tensors='pt')
    try:
        with torch.inference_mode():
            response = model.generate(
                pixel_values=inputs['pixel_values'].to(device='cuda', dtype=torch.bfloat16),
                input_ids=inputs['input_ids'].cuda(),
                attention_mask=inputs['attention_mask'].cuda(),
                image_grid_hws=torch.as_tensor(inputs['image_grid_hws'], device='cuda'),
                tokenizer=tokenizer, use_cache=True, max_new_tokens=max_new_tokens,
                generation_mode='hybrid', do_sample=temperature > 0,
                temperature=temperature, top_p=0.9, repetition_penalty=1.1)
        return answer_to_text(response, tokenizer)
    finally:
        del inputs
        gc.collect()
        torch.cuda.empty_cache()


def infer_one(key, item, reference, args, model, tokenizer, processor):
    from PIL import Image

    rank, direction, target = parse_ordinal(item['query'])
    seed = args.seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    with Image.open(args.data_root / item['visible']) as source:
        image = source.convert('RGB')
    # This is the model card's ground_multi prompt, not an enumerated single-box query.
    multi_prompt = f'Locate all the instances that match the following description: {target}.'
    raw_answer = generate(model, tokenizer, processor, image, multi_prompt,
                          args.multi_max_new_tokens, seed, args.temperature,
                          args.image_token_limit)
    raw = parse_boxes(raw_answer)
    no_parents = remove_parent_boxes(raw, args.parent_containment, args.parent_min_children,
                                     args.parent_min_area_ratio, args.dedup_iou)
    unique = deduplicate(no_parents, args.dedup_iou)
    ordered = sorted(unique, key=lambda box: (box[0] + box[2]) / 2)
    result = dict(id=key, status='no_candidates', bbox=None, query=item['query'],
                  visible=item['visible'], rank=rank, direction=direction, target=target,
                  multi_prompt=multi_prompt, multi_answer=raw_answer, raw_boxes=raw,
                  after_parent=no_parents, after_dedup=ordered, selection_answer='',
                  puzzle_box=None, selected_index=None, puzzle_size=None)
    if not ordered:
        return result
    puzzle, spans, used = make_puzzle(image, ordered, args.tile_height, args.gap,
                                     args.max_puzzle_width)
    if puzzle is None:
        result['status'] = 'invalid_puzzle'
        return result
    result['puzzle_size'] = list(puzzle.size)
    direction_text = direction.replace('_', ' ')
    word = ORDINAL_WORDS.get(rank, f'{rank}th')
    select_prompt = ('Locate a single instance that matches the following description: '
                     f'the {word} {target} from {direction_text} among the separate outlined items.')
    result['selection_prompt'] = select_prompt
    selection_answer = generate(model, tokenizer, processor, puzzle, select_prompt,
                                args.select_max_new_tokens, seed + 1, args.temperature,
                                args.image_token_limit)
    result['selection_answer'] = selection_answer
    puzzle_boxes = parse_boxes(selection_answer)
    if puzzle_boxes:
        result['puzzle_box'] = puzzle_boxes[0]
        selected = select_tile(puzzle_boxes[0], spans)
        result['selected_index'] = selected
        if selected is not None:
            result['bbox'] = used[selected]
            result['status'] = 'ok'
    if result['status'] != 'ok':
        result['status'] = 'selection_failed'
    draw_puzzle_result(puzzle, spans, used, reference['bbox'], result['selected_index'],
                       args.output_dir / 'puzzles' / f'{key}.png')
    return result


def summarize(rows, references, records, baseline, output):
    details, predictions = [], {}
    for key, item in rows:
        ref = references[key]['bbox']
        record = records.get(key, {})
        box = record.get('bbox') if record.get('status') == 'ok' else None
        iou = box_iou(box, ref) if valid_box(box) else 0.0
        if valid_box(box):
            predictions[key] = {**item, 'bbox': box}
        baseline_row = baseline.get(key, {})
        baseline_box = baseline_row.get('bbox') if isinstance(baseline_row, dict) else None
        baseline_iou = box_iou(baseline_box, ref) if valid_box(baseline_box) else 0.0
        raw, parent, unique = (record.get(name, []) for name in
                               ('raw_boxes', 'after_parent', 'after_dedup'))
        details.append(dict(id=key, status=record.get('status', 'pending'), query=item['query'],
                            target=record.get('target', ''), rank=record.get('rank', ''),
                            direction=record.get('direction', ''), raw_count=len(raw),
                            after_parent_count=len(parent), after_dedup_count=len(unique),
                            raw_gt_count=gt_count(raw, ref), parent_gt_count=gt_count(parent, ref),
                            dedup_gt_count=gt_count(unique, ref), selected_index=record.get('selected_index'),
                            f_iou=iou, a_iou=baseline_iou if baseline else None,
                            error=record.get('error', '')))
    scored = [row['f_iou'] for row in details]
    processed = sum(row['status'] != 'pending' for row in details)
    summary = dict(total=len(rows), processed=processed, successful=len(predictions),
                   failed=processed - len(predictions), pending=len(rows) - processed,
                   mean_iou=statistics.mean(scored),
                   acc_at_05=sum(score >= .5 for score in scored) / len(rows),
                   gt_in_raw=sum(row['raw_gt_count'] > 0 for row in details),
                   gt_after_parent=sum(row['parent_gt_count'] > 0 for row in details),
                   gt_after_dedup=sum(row['dedup_gt_count'] > 0 for row in details),
                   denominator='all selected ordinal references; missing/failed predictions score zero')
    if baseline:
        summary['baseline_valid_boxes'] = sum(
            valid_box(baseline[key].get('bbox')) for key, _ in rows
            if key in baseline and isinstance(baseline[key], dict))
        summary['baseline_mean_iou'] = statistics.mean(row['a_iou'] for row in details)
        summary['baseline_acc_at_05'] = sum(row['a_iou'] >= .5 for row in details) / len(rows)
        summary['a_correct_f_correct'] = sum(row['a_iou'] >= .5 and row['f_iou'] >= .5 for row in details)
        summary['a_correct_f_wrong'] = sum(row['a_iou'] >= .5 and row['f_iou'] < .5 for row in details)
        summary['a_wrong_f_correct'] = sum(row['a_iou'] < .5 and row['f_iou'] >= .5 for row in details)
        summary['a_wrong_f_wrong'] = sum(row['a_iou'] < .5 and row['f_iou'] < .5 for row in details)
    atomic_json(output / 'queries_f.json', predictions)
    atomic_json(output / 'summary.json', summary)
    with (output / 'per_query.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    return summary


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=root / 'datasets/ordinal_subset/queries.json')
    parser.add_argument('--references', type=Path, default=root / 'datasets/ordinal_subset/annotations.json')
    parser.add_argument('--data-root', type=Path, default=root / 'datasets/reference_subset')
    parser.add_argument('--model-path', type=Path, default=root / 'LocateAnything-3B')
    parser.add_argument('--output-dir', type=Path, default=root / 'outputs/ordinal_f_official_multi')
    parser.add_argument('--baseline-predictions', type=Path)
    parser.add_argument('--limit', type=int, default=0, help='0 means all ordinal queries')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--image-token-limit', type=int, default=4096)
    parser.add_argument('--multi-max-new-tokens', type=int, default=2048)
    parser.add_argument('--select-max-new-tokens', type=int, default=128)
    parser.add_argument('--parent-containment', type=float, default=0.9)
    parser.add_argument('--parent-min-children', type=int, default=2)
    parser.add_argument('--parent-min-area-ratio', type=float, default=1.5)
    parser.add_argument('--dedup-iou', type=float, default=0.5)
    parser.add_argument('--tile-height', type=int, default=224)
    parser.add_argument('--gap', type=int, default=12)
    parser.add_argument('--max-puzzle-width', type=int, default=1536)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if (args.limit < 0 or args.seed < 0 or args.temperature < 0 or args.parent_min_children < 1
            or args.parent_min_area_ratio <= 1 or args.gap < 0
            or min(args.image_token_limit, args.multi_max_new_tokens, args.select_max_new_tokens,
                   args.tile_height, args.max_puzzle_width) < 1
            or not 0 < args.parent_containment <= 1 or not 0 <= args.dedup_iou <= 1):
        parser.error('Invalid numeric arguments')
    args.data_root = args.data_root.resolve()
    args.output_dir = args.output_dir.resolve()
    queries, references = load_json(args.queries), load_json(args.references)
    if not queries or set(queries) != set(references):
        raise ValueError('Queries/references must be nonempty and have identical IDs')
    rows = list(queries.items())[:args.limit or None]
    for key, item in rows:
        if parse_ordinal(item['query']) is None or item['query'] != references[key].get('query'):
            raise ValueError(f'{key}: not an explicit horizontal ordinal or reference mismatch')
        if not valid_box(references[key].get('bbox')):
            raise ValueError(f'{key}: invalid reference bbox')
        relative = Path(item['visible'])
        if relative.is_absolute() or '..' in relative.parts or not (args.data_root / relative).is_file():
            raise ValueError(f'{key}: missing/unsafe RGB image {relative}')
    baseline = load_json(args.baseline_predictions) if args.baseline_predictions else {}
    print(f'Validated {len(rows)} of {len(queries)} ordinal queries', flush=True)
    if args.check_only:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / 'run.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        config = dict(script_sha256=sha256(Path(__file__)),
                      queries_sha256=sha256(args.queries), references_sha256=sha256(args.references),
                      baseline_sha256=sha256(args.baseline_predictions) if args.baseline_predictions else None,
                      data_root=str(args.data_root), model_path=str(args.model_path.resolve()),
                      limit=args.limit, seed=args.seed, temperature=args.temperature,
                      image_token_limit=args.image_token_limit,
                      multi_max_new_tokens=args.multi_max_new_tokens,
                      select_max_new_tokens=args.select_max_new_tokens,
                      parent_containment=args.parent_containment,
                      parent_min_children=args.parent_min_children,
                      parent_min_area_ratio=args.parent_min_area_ratio, dedup_iou=args.dedup_iou,
                      tile_height=args.tile_height, gap=args.gap, max_puzzle_width=args.max_puzzle_width)
        manifest = args.output_dir / 'run_config.json'
        if manifest.exists():
            if json.loads(manifest.read_text(encoding='utf-8')) != config:
                raise ValueError('Run configuration changed; choose a new --output-dir')
        else:
            if (args.output_dir / 'predictions.jsonl').exists():
                raise ValueError('Existing journal without run_config.json')
            atomic_json(manifest, config)
        records = read_progress(args.output_dir / 'predictions.jsonl')
        model = tokenizer = processor = None
        with (args.output_dir / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
            for index, (key, item) in enumerate(rows, 1):
                if records.get(key, {}).get('status') in ('ok', 'no_candidates', 'selection_failed', 'invalid_puzzle'):
                    continue
                if model is None:
                    model, tokenizer, processor = load_model(args.model_path, args.image_token_limit)
                try:
                    record = infer_one(key, item, references[key], args, model, tokenizer, processor)
                except Exception as exc:
                    record = dict(id=key, status='error', bbox=None,
                                  error=f'{type(exc).__name__}: {exc}')
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                records[key] = record
                summary = summarize(rows, references, records, baseline, args.output_dir)
                print(f'{index}/{len(rows)} {key}: {record["status"]}; '
                      f'F Acc@0.5={summary["acc_at_05"]:.3f}', flush=True)
        summary = summarize(rows, references, records, baseline, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary['pending'] == 0 and summary['failed'] == 0 else 2
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
