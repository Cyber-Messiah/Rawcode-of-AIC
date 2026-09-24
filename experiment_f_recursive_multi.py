#!/usr/bin/env python3
"""Refine oversized multi-grounding boxes before ordinal puzzle selection.

The reference boxes are read only by summarize(), after inference. The original
F answer, refined answer, and final answer are all recorded for comparison.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from PIL import ImageDraw

from evaluate import box_iou
from experiment_f_official_multi import (
    ORDINAL_WORDS, area, deduplicate, generate, intersection, load_model,
    make_puzzle, parse_boxes, remove_parent_boxes, select_tile,
)
from predict_rgb_all import atomic_json, read_progress
from prepare_ordinal_subset import parse_ordinal
from prepare_reference_subset import load_json, sha256, valid_box


def suspicious_parent(boxes, single_area=0.10, multi_area=0.10,
                      relative_area=1.75, baseline_box=None,
                      a_area_ratio=3.0, a_containment=0.9,
                      a_min_area=0.02):
    """Return (index, reasons) for the largest box, or (None, []).

    All areas are fractions of the current input image. The A comparison is
    available only for the original image, where both boxes share coordinates.
    """
    if not boxes:
        return None, []
    ordered = sorted(range(len(boxes)), key=lambda i: area(boxes[i]), reverse=True)
    largest = boxes[ordered[0]]
    reasons = []
    if len(boxes) == 1:
        if area(largest) >= single_area:
            reasons.append('single_absolute_area')
    elif area(largest) >= multi_area and area(largest) / area(boxes[ordered[1]]) >= relative_area:
        reasons.append('relative_area_outlier')
    if (baseline_box is not None and valid_box(baseline_box)
            and area(largest) >= a_min_area
            and intersection(largest, baseline_box) / area(baseline_box) >= a_containment
            and area(largest) / area(baseline_box) >= a_area_ratio):
        reasons.append('encloses_smaller_a_box')
    return (ordered[0], reasons) if reasons else (None, [])


def crop_region(image, box, padding=0.02):
    """Return a padded crop and its integer xyxy bounds in the original image."""
    width, height = image.size
    pad_x = (box[2] - box[0]) * padding
    pad_y = (box[3] - box[1]) * padding
    left = max(0, min(width - 1, math.floor((box[0] - pad_x) * width)))
    top = max(0, min(height - 1, math.floor((box[1] - pad_y) * height)))
    right = max(left + 1, min(width, math.ceil((box[2] + pad_x) * width)))
    bottom = max(top + 1, min(height, math.ceil((box[3] + pad_y) * height)))
    return image.crop((left, top, right, bottom)), (left, top, right, bottom)


def map_from_crop(box, bounds, image_size):
    """Map a normalized crop box into normalized full-image coordinates."""
    left, top, right, bottom = bounds
    width, height = image_size
    mapped = [(left + box[0] * (right - left)) / width,
              (top + box[1] * (bottom - top)) / height,
              (left + box[2] * (right - left)) / width,
              (top + box[3] * (bottom - top)) / height]
    return [max(0.0, min(1.0, value)) for value in mapped]


def effective_children(local_boxes, bounds, image_size, parent,
                       max_area_ratio=0.8, min_containment=0.9, dedup_iou=0.5):
    """Keep meaningfully smaller child boxes inside the suspected parent."""
    accepted = []
    for local in deduplicate(local_boxes, dedup_iou):
        child = map_from_crop(local, bounds, image_size)
        if (valid_box(child) and area(child) <= area(parent) * max_area_ratio
                and intersection(parent, child) / area(child) >= min_containment):
            accepted.append((local, child))
    return accepted


def replace_parent(active, parent, children, dedup_iou):
    """Use children as active instances; reserve the parent outside the puzzle."""
    position = next((i for i, box in enumerate(active) if box_iou(box, parent) >= 0.999), None)
    if position is None:
        return deduplicate(active + children, dedup_iou)
    return deduplicate(active[:position] + children + active[position + 1:], dedup_iou)


def select_on_puzzle(image, boxes, rank, direction, target, args,
                     model, tokenizer, processor, seed, picture_path):
    from PIL import Image

    if not boxes:
        return dict(bbox=None, answer='', puzzle_box=None, selected_index=None,
                    puzzle_size=None, status='no_candidates')
    ordered = sorted(boxes, key=lambda box: (box[0] + box[2]) / 2)
    puzzle, spans, used = make_puzzle(image, ordered, args.tile_height,
                                      args.gap, args.max_puzzle_width)
    if puzzle is None:
        return dict(bbox=None, answer='', puzzle_box=None, selected_index=None,
                    puzzle_size=None, status='invalid_puzzle')
    word = ORDINAL_WORDS.get(rank, f'{rank}th')
    prompt = ('Locate a single instance that matches the following description: '
              f'the {word} {target} from {direction.replace("_", " ")} '
              'among the separate outlined items.')
    answer = generate(model, tokenizer, processor, puzzle, prompt,
                      args.select_max_new_tokens, seed + 1, args.temperature,
                      args.image_token_limit)
    puzzle_boxes = parse_boxes(answer)
    selected = select_tile(puzzle_boxes[0], spans) if puzzle_boxes else None
    if picture_path is not None:
        marked = puzzle.copy()
        if selected is not None:
            draw = ImageDraw.Draw(marked)
            x1 = round(spans[selected][0] * marked.width)
            x2 = round(spans[selected][1] * marked.width) - 1
            draw.rectangle((x1, 0, x2, marked.height - 1), outline='red', width=4)
        picture_path.parent.mkdir(parents=True, exist_ok=True)
        marked.save(picture_path)
    return dict(bbox=used[selected] if selected is not None else None,
                answer=answer, puzzle_box=puzzle_boxes[0] if puzzle_boxes else None,
                selected_index=selected, puzzle_size=list(puzzle.size),
                status='ok' if selected is not None else 'selection_failed', prompt=prompt,
                candidate_count=len(used))


def infer_one(key, item, baseline_box, args, model, tokenizer, processor):
    from PIL import Image

    rank, direction, target = parse_ordinal(item['query'])
    seed = args.seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    with Image.open(args.data_root / item['visible']) as source:
        image = source.convert('RGB')
    multi_prompt = f'Locate all the instances that match the following description: {target}.'
    multi_answer = generate(model, tokenizer, processor, image, multi_prompt,
                            args.multi_max_new_tokens, seed, args.temperature,
                            args.image_token_limit)
    raw = parse_boxes(multi_answer)
    root_unique = deduplicate(raw, args.dedup_iou)
    active = deduplicate(remove_parent_boxes(raw, args.parent_containment,
                              args.parent_min_children, args.parent_min_area_ratio,
                              args.dedup_iou), args.dedup_iou)
    initial = select_on_puzzle(image, active, rank, direction, target, args,
                               model, tokenizer, processor, seed,
                               args.output_dir / 'puzzles_initial' / f'{key}.png' if active else None)
    result = dict(id=key, status=initial['status'], bbox=initial['bbox'],
                  query=item['query'], visible=item['visible'], rank=rank,
                  direction=direction, target=target, multi_prompt=multi_prompt,
                  multi_answer=multi_answer, raw_boxes=raw, original_boxes=active,
                  original_selection=initial, refinement_steps=[],
                  refined_boxes=active, refined_selection=None,
                  final_source='original_f')
    if not raw:
        if valid_box(baseline_box):
            result.update(status='fallback_a_none', bbox=baseline_box, final_source='a_none')
        return result

    current_local = root_unique
    current_bounds = (0, 0, image.width, image.height)
    for depth in range(1, args.max_depth + 1):
        a_box = baseline_box if depth == 1 else None
        index, reasons = suspicious_parent(
            current_local, args.single_area_threshold, args.multi_area_threshold,
            args.relative_area_ratio, a_box, args.a_area_ratio, args.a_containment,
            args.a_min_area_threshold)
        if index is None:
            break
        local_parent = current_local[index]
        parent = map_from_crop(local_parent, current_bounds, image.size)
        # A crop can make a small original-image box look large locally.
        if depth > 1 and area(parent) < args.recursive_global_area_threshold:
            break
        crop, bounds = crop_region(image, parent, args.crop_padding)
        step = dict(depth=depth, reasons=reasons, parent=parent,
                    crop_bounds=list(bounds), crop_size=list(crop.size),
                    local_area=area(local_parent))
        try:
            answer = generate(model, tokenizer, processor, crop, multi_prompt,
                              args.multi_max_new_tokens, seed + 100 * depth,
                              args.temperature, args.image_token_limit)
            children_local = parse_boxes(answer)
            children = effective_children(children_local, bounds, image.size, parent,
                                          args.child_max_area_ratio,
                                          args.child_min_containment, args.dedup_iou)
            step.update(answer=answer, raw_child_count=len(children_local),
                        accepted_children=[global_box for _, global_box in children])
        except Exception as exc:
            step['error'] = f'{type(exc).__name__}: {exc}'
            result['refinement_steps'].append(step)
            break
        result['refinement_steps'].append(step)
        if not children:
            break
        active = replace_parent(active, parent, [global_box for _, global_box in children],
                                args.dedup_iou)
        current_local = [local for local, global_box in children
                         if any(box_iou(global_box, survivor) >= 0.999 for survivor in active)]
        if not current_local:
            break
        current_bounds = bounds

    result['refined_boxes'] = active
    if any(step.get('accepted_children') for step in result['refinement_steps']):
        refined = select_on_puzzle(image, active, rank, direction, target, args,
                                   model, tokenizer, processor, seed,
                                   args.output_dir / 'puzzles_refined' / f'{key}.png')
        result['refined_selection'] = refined
        if valid_box(refined['bbox']):
            result.update(status='ok', bbox=refined['bbox'], final_source='refined_f')
    return result


def summarize(rows, references, records, baseline, output):
    details, predictions = [], {}
    for key, item in rows:
        reference = references[key]['bbox']
        record = records.get(key, {})
        final_box = record.get('bbox') if record.get('status') in ('ok', 'fallback_a_none') else None
        initial_box = record.get('original_selection', {}).get('bbox')
        refined = record.get('refined_selection') or {}
        baseline_box = baseline.get(key, {}).get('bbox')
        score = lambda box: box_iou(box, reference) if valid_box(box) else 0.0
        if valid_box(final_box):
            predictions[key] = {**item, 'bbox': final_box}
        steps = record.get('refinement_steps', [])
        raw, refined_boxes = record.get('raw_boxes', []), record.get('refined_boxes', [])
        details.append(dict(id=key, status=record.get('status', 'pending'),
                            final_source=record.get('final_source', ''), query=item['query'],
                            target=record.get('target', ''), raw_count=len(raw),
                            original_count=len(record.get('original_boxes', [])),
                            refined_count=len(refined_boxes), refinement_calls=len(steps),
                            successful_refinements=sum(bool(s.get('accepted_children')) for s in steps),
                            raw_gt_hit=any(score(b) >= .5 for b in raw),
                            refined_gt_hit=any(score(b) >= .5 for b in refined_boxes),
                            initial_f_iou=score(initial_box), refined_f_iou=score(refined.get('bbox')),
                            final_iou=score(final_box), a_iou=score(baseline_box),
                            error=record.get('error', '')))
    total = len(details)
    processed = sum(row['status'] != 'pending' for row in details)
    acc = lambda field: sum(row[field] >= .5 for row in details) / total
    mean = lambda field: statistics.mean(row[field] for row in details)
    summary = dict(total=total, processed=processed, pending=total - processed,
                   successful=len(predictions), failed=processed - len(predictions),
                   final_mean_iou=mean('final_iou'), final_acc_at_05=acc('final_iou'),
                   initial_f_mean_iou=mean('initial_f_iou'),
                   initial_f_acc_at_05=acc('initial_f_iou'),
                   baseline_mean_iou=mean('a_iou'), baseline_acc_at_05=acc('a_iou'),
                   explicit_none=sum('<box>None</box>' in records.get(key, {}).get('multi_answer', '')
                                     for key, _ in rows),
                   a_none_fallbacks=sum(row['final_source'] == 'a_none' for row in details),
                   refinement_triggered=sum(row['refinement_calls'] > 0 for row in details),
                   refinement_calls=sum(row['refinement_calls'] for row in details),
                   refinement_accepted=sum(row['successful_refinements'] for row in details),
                   raw_gt_hit=sum(row['raw_gt_hit'] for row in details),
                   refined_gt_hit=sum(row['refined_gt_hit'] for row in details),
                   new_gt_recovered=sum(not row['raw_gt_hit'] and row['refined_gt_hit'] for row in details),
                   gt_lost=sum(row['raw_gt_hit'] and not row['refined_gt_hit'] for row in details),
                   refinement_new_correct=sum(row['refinement_calls'] > 0 and
                                              row['initial_f_iou'] < .5 and row['final_iou'] >= .5
                                              for row in details),
                   refinement_lost_correct=sum(row['refinement_calls'] > 0 and
                                               row['initial_f_iou'] >= .5 and row['final_iou'] < .5
                                               for row in details),
                   a_correct_final_correct=sum(row['a_iou'] >= .5 and row['final_iou'] >= .5 for row in details),
                   a_correct_final_wrong=sum(row['a_iou'] >= .5 and row['final_iou'] < .5 for row in details),
                   a_wrong_final_correct=sum(row['a_iou'] < .5 and row['final_iou'] >= .5 for row in details),
                   a_wrong_final_wrong=sum(row['a_iou'] < .5 and row['final_iou'] < .5 for row in details),
                   denominator='all selected references; missing/invalid predictions count as IoU 0')
    atomic_json(output / 'queries_recursive.json', predictions)
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
    parser.add_argument('--baseline-predictions', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=root / 'outputs/ordinal_f_recursive')
    parser.add_argument('--ids', nargs='*', default=None)
    parser.add_argument('--limit', type=int, default=0)
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
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--single-area-threshold', type=float, default=0.10)
    parser.add_argument('--multi-area-threshold', type=float, default=0.10)
    parser.add_argument('--relative-area-ratio', type=float, default=1.75)
    parser.add_argument('--a-area-ratio', type=float, default=3.0)
    parser.add_argument('--a-containment', type=float, default=0.9)
    parser.add_argument('--a-min-area-threshold', type=float, default=0.02)
    parser.add_argument('--recursive-global-area-threshold', type=float, default=0.10)
    parser.add_argument('--child-max-area-ratio', type=float, default=0.8)
    parser.add_argument('--child-min-containment', type=float, default=0.9)
    parser.add_argument('--crop-padding', type=float, default=0.02)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if (args.limit < 0 or args.seed < 0 or args.temperature < 0 or args.max_depth < 0
            or args.parent_min_children < 1 or args.parent_min_area_ratio <= 1
            or min(args.image_token_limit, args.multi_max_new_tokens, args.select_max_new_tokens,
                   args.tile_height, args.max_puzzle_width) < 1 or args.gap < 0
            or not 0 < args.single_area_threshold <= 1
            or not 0 < args.multi_area_threshold <= 1
            or args.relative_area_ratio <= 1 or args.a_area_ratio <= 1
            or not 0 < args.a_min_area_threshold <= 1
            or not 0 < args.recursive_global_area_threshold <= 1
            or not 0 < args.a_containment <= 1 or not 0 < args.parent_containment <= 1
            or not 0 < args.child_max_area_ratio < 1
            or not 0 < args.child_min_containment <= 1
            or not 0 <= args.dedup_iou <= 1 or not 0 <= args.crop_padding <= 0.2):
        parser.error('Invalid numeric arguments')
    queries, references = load_json(args.queries), load_json(args.references)
    baseline = load_json(args.baseline_predictions)
    if not queries or set(queries) != set(references):
        raise ValueError('Queries/references must be nonempty and have identical IDs')
    if args.ids is not None:
        unknown = set(args.ids) - set(queries)
        if unknown:
            raise ValueError(f'Unknown query IDs: {sorted(unknown)}')
        rows = [(key, queries[key]) for key in dict.fromkeys(args.ids)]
    else:
        rows = list(queries.items())[:args.limit or None]
    if not rows:
        raise ValueError('No selected queries')
    for key, item in rows:
        if (parse_ordinal(item['query']) is None
                or item['query'] != references[key].get('query')
                or not valid_box(references[key].get('bbox'))):
            raise ValueError(f'{key}: invalid ordinal reference')
        if not valid_box(baseline.get(key, {}).get('bbox')):
            raise ValueError(f'{key}: missing/invalid A baseline bbox')
        visible = Path(item['visible'])
        if visible.is_absolute() or '..' in visible.parts or not (args.data_root / visible).is_file():
            raise ValueError(f'{key}: missing/unsafe RGB image')
    print(f'Validated {len(rows)} of {len(queries)} ordinal queries', flush=True)
    if args.check_only:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / 'run.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        config = dict(script_sha256=sha256(Path(__file__)),
                      base_script_sha256=sha256(Path(__file__).with_name('experiment_f_official_multi.py')),
                      parser_sha256=sha256(Path(__file__).with_name('prepare_ordinal_subset.py')),
                      queries_sha256=sha256(args.queries), references_sha256=sha256(args.references),
                      baseline_sha256=sha256(args.baseline_predictions),
                      data_root=str(args.data_root.resolve()), model_path=str(args.model_path.resolve()),
                      selected_ids=[key for key, _ in rows],
                      settings={name:getattr(args,name) for name in (
                          'seed','temperature','image_token_limit','multi_max_new_tokens',
                          'select_max_new_tokens','parent_containment','parent_min_children',
                          'parent_min_area_ratio','dedup_iou','tile_height','gap',
                          'max_puzzle_width','max_depth','single_area_threshold',
                          'multi_area_threshold','relative_area_ratio','a_area_ratio',
                          'a_containment','a_min_area_threshold',
                          'recursive_global_area_threshold','child_max_area_ratio',
                          'child_min_containment','crop_padding')})
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
                if records.get(key, {}).get('status') in ('ok','fallback_a_none','no_candidates',
                                                           'selection_failed','invalid_puzzle'):
                    continue
                if model is None:
                    model, tokenizer, processor = load_model(args.model_path, args.image_token_limit)
                try:
                    record = infer_one(key, item, baseline[key]['bbox'], args,
                                       model, tokenizer, processor)
                except Exception as exc:
                    record = dict(id=key, status='error', bbox=None,
                                  error=f'{type(exc).__name__}: {exc}')
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                records[key] = record
                summary = summarize(rows, references, records, baseline, args.output_dir)
                print(f'{index}/{len(rows)} {key}: {record["status"]}, '
                      f'depth={len(record.get("refinement_steps", []))}, '
                      f'final Acc@0.5={summary["final_acc_at_05"]:.3f}', flush=True)
        summary = summarize(rows, references, records, baseline, args.output_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary['pending'] == 0 and summary['failed'] == 0 else 2
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
