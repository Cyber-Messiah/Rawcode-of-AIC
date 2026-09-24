#!/usr/bin/env python3
"""Re-query unresolved very wide F boxes, then evaluate the resulting G candidates.

This is a separate G experiment. It reads an existing F journal, never reads GT,
and writes a new journal whose ``refined_boxes`` may contain accepted splits.
Run evaluate_g_ordinal.py against the new journal with --candidate-stage refined.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image

from evaluate import box_iou
from experiment_f_official_multi import deduplicate, generate, load_model, parse_boxes
from experiment_f_recursive_multi import crop_region, map_from_crop, replace_parent
from predict_rgb_all import atomic_json, read_progress
from prepare_reference_subset import sha256, valid_box


def width(box):
    return box[2] - box[0]


def area(box):
    return width(box) * (box[3] - box[1])


def wide_indices(boxes, threshold):
    """All still-wide candidates, widest first; independent of box area."""
    return sorted((i for i, box in enumerate(boxes) if width(box) >= threshold),
                  key=lambda i: width(boxes[i]), reverse=True)


def horizontal_tiles(image, parent, fraction=0.42, overlap=0.15, padding=0.02):
    """Cover a wide parent with overlapping vertical crops in full-image pixels."""
    _, (left, top, right, bottom) = crop_region(image, parent, padding)
    span = right - left
    tile_width = max(1, min(span, math.ceil(span * fraction)))
    if tile_width == span:
        return [(image.crop((left, top, right, bottom)), (left, top, right, bottom))]
    stride = max(1, round(tile_width * (1 - overlap)))
    starts = list(range(left, right - tile_width + 1, stride))
    last = right - tile_width
    if not starts or starts[-1] != last:
        starts.append(last)
    return [(image.crop((x, top, x + tile_width, bottom)),
             (x, top, x + tile_width, bottom)) for x in starts]


def strict_children(local_boxes, bounds, image_size, parent, max_width_ratio,
                    max_area_ratio, min_containment, dedup_iou,
                    local_width_limit=1.0):
    """Reject whole-crop outputs and children that still span much of a parent."""
    accepted = []
    for local in local_boxes:
        if width(local) >= local_width_limit:
            continue
        child = map_from_crop(local, bounds, image_size)
        if (valid_box(child)
                and width(child) <= width(parent) * max_width_ratio
                and area(child) <= area(parent) * max_area_ratio
                and box_iou(child, parent) > 0):
            left = max(child[0], parent[0])
            top = max(child[1], parent[1])
            right = min(child[2], parent[2])
            bottom = min(child[3], parent[3])
            overlap_area = max(0, right - left) * max(0, bottom - top)
            if overlap_area / area(child) >= min_containment:
                accepted.append(child)
    return deduplicate(accepted, dedup_iou)


def enough_children(active, parent, children, rank, dedup_iou):
    """A wide parent must become multiple distinct, countable instances."""
    if len(children) < 2:
        return False
    replaced = replace_parent(active, parent, children, dedup_iou)
    return len(replaced) >= rank and len(replaced) > len(active)


def split_wide_parent(image, active, parent, rank, target, key_seed, args,
                      model, tokenizer, processor):
    crop, bounds = crop_region(image, parent, args.crop_padding)
    prompts = [
        (f'Locate every separate individual {target} visible in this image. '
         'Return one tight box per individual instance. Never return a box '
         'covering a group, a row, or the whole image.'),
        (f'Count the distinct {target} instances from left to right. '
         'For each instance, output its own small, tight bounding box. '
         'A box containing two or more instances is invalid. '
         'Do not output an enclosing group box.'),
    ]
    attempts, collected = [], []
    call_number = 0

    def attempt(picture, picture_bounds, prompt, kind, local_width_limit=1.0):
        nonlocal call_number, collected
        call_number += 1
        step = dict(kind=kind, prompt=prompt, bounds=list(picture_bounds))
        try:
            answer = generate(model, tokenizer, processor, picture, prompt,
                              args.multi_max_new_tokens, key_seed + 1000 + call_number,
                              args.temperature, args.image_token_limit)
            raw = parse_boxes(answer)
            accepted = strict_children(raw, picture_bounds, image.size, parent,
                                       args.wide_child_max_width_ratio,
                                       args.wide_child_max_area_ratio,
                                       args.wide_child_min_containment,
                                       args.wide_dedup_iou, local_width_limit)
            step.update(answer=answer, raw_child_count=len(raw), accepted_children=accepted)
            collected = deduplicate(collected + accepted, args.wide_dedup_iou)
        except Exception as exc:
            step["error"] = f'{type(exc).__name__}: {exc}'
        attempts.append(step)
        return enough_children(active, parent, collected, rank, args.wide_dedup_iou)

    for index, prompt in enumerate(prompts, 1):
        if attempt(crop, bounds, prompt, f'full_crop_prompt_{index}'):
            return collected, attempts

    tile_prompt = (f'Locate each separate {target} in this cropped image. '
                   'Return one tight box per individual object. '
                   'Ignore objects not visible in this crop; never box the whole crop.')
    for index, (tile, tile_bounds) in enumerate(horizontal_tiles(
            image, parent, args.wide_tile_fraction, args.wide_tile_overlap,
            args.crop_padding), 1):
        if attempt(tile, tile_bounds, tile_prompt, f'horizontal_tile_{index}',
                   args.wide_tile_max_local_width):
            return collected, attempts
    return [], attempts


def refine_record(record, args, model, tokenizer, processor):
    """Apply the width gate to every residual wide candidate in an F record."""
    result = dict(record)
    active = [list(box) for box in (record.get('refined_boxes') or [])]
    pending = [active[i] for i in wide_indices(active, args.wide_width_threshold)]
    steps = []
    if not pending:
        return result
    relative = Path(record['visible'])
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'{record["id"]}: unsafe image path')
    with Image.open(args.data_root / relative) as source:
        image = source.convert('RGB')
    key_seed = args.seed + int(hashlib.sha256(record['id'].encode()).hexdigest()[:8], 16)
    for index, parent in enumerate(pending):
        if not any(box_iou(parent, box) >= .999 for box in active):
            continue
        children, attempts = split_wide_parent(
            image, active, parent, record['rank'], record['target'],
            key_seed + index * 100, args, model, tokenizer, processor)
        accepted = bool(children)
        if accepted:
            active = replace_parent(active, parent, children, args.wide_dedup_iou)
        steps.append(dict(parent=parent, parent_width=width(parent),
                          accepted=accepted, children=children, attempts=attempts))
    result['wide_refinement_steps'] = steps
    result['refined_boxes'] = active
    return result


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--ids', nargs='*', help='Only re-query these IDs; other journal rows are copied unchanged')
    parser.add_argument('--data-root', type=Path, default=root / 'datasets/reference_subset')
    parser.add_argument('--model-path', type=Path, default=root / 'LocateAnything-3B')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=0.4)
    parser.add_argument('--image-token-limit', type=int, default=4096)
    parser.add_argument('--multi-max-new-tokens', type=int, default=2048)
    parser.add_argument('--crop-padding', type=float, default=0.02)
    parser.add_argument('--wide-width-threshold', type=float, default=0.8)
    parser.add_argument('--wide-child-max-width-ratio', type=float, default=0.6)
    parser.add_argument('--wide-child-max-area-ratio', type=float, default=0.45)
    parser.add_argument('--wide-child-min-containment', type=float, default=0.9)
    parser.add_argument('--wide-dedup-iou', type=float, default=0.4)
    parser.add_argument('--wide-tile-fraction', type=float, default=0.42)
    parser.add_argument('--wide-tile-overlap', type=float, default=0.15)
    parser.add_argument('--wide-tile-max-local-width', type=float, default=0.85)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if (args.seed < 0 or args.temperature < 0 or args.image_token_limit < 1
            or args.multi_max_new_tokens < 1 or not 0 <= args.crop_padding <= 0.2
            or not 0 < args.wide_width_threshold <= 1
            or not 0 < args.wide_child_max_width_ratio < 1
            or not 0 < args.wide_child_max_area_ratio < 1
            or not 0 < args.wide_child_min_containment <= 1
            or not 0 <= args.wide_dedup_iou <= 1
            or not 0 < args.wide_tile_fraction < 1
            or not 0 <= args.wide_tile_overlap < 1
            or not 0 < args.wide_tile_max_local_width < 1):
        parser.error('Invalid numeric arguments')
    records = read_progress(args.journal)
    if not records:
        raise ValueError('Empty F journal')
    if args.ids is not None:
        unknown = set(args.ids) - set(records)
        if unknown:
            raise ValueError(f'Unknown IDs: {sorted(unknown)}')
    selected = {key: [r['refined_boxes'][i] for i in wide_indices(
        r.get('refined_boxes') or [], args.wide_width_threshold)]
        for key, r in records.items()}
    selected = {key: boxes for key, boxes in selected.items()
                if boxes and (args.ids is None or key in args.ids)}
    print(json.dumps(dict(total=len(records), triggered=len(selected),
                          ids=list(selected), threshold=args.wide_width_threshold),
                     ensure_ascii=False, indent=2), flush=True)
    if args.check_only:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(script_sha256=sha256(Path(__file__)), journal_sha256=sha256(args.journal),
                  data_root=str(args.data_root.resolve()), model_path=str(args.model_path.resolve()),
                  ids=args.ids,
                  settings={name: getattr(args, name) for name in (
                      'seed', 'temperature', 'image_token_limit', 'multi_max_new_tokens',
                      'crop_padding', 'wide_width_threshold', 'wide_child_max_width_ratio',
                      'wide_child_max_area_ratio', 'wide_child_min_containment',
                      'wide_dedup_iou', 'wide_tile_fraction', 'wide_tile_overlap',
                      'wide_tile_max_local_width')})
    manifest = args.output_dir / 'run_config.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != config:
            raise ValueError('Run configuration changed; choose a new --output-dir')
    else:
        if (args.output_dir / 'predictions.jsonl').exists():
            raise ValueError('Existing journal without run_config.json')
        atomic_json(manifest, config)
    lock = args.output_dir / 'run.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        completed = read_progress(args.output_dir / 'predictions.jsonl')
        model = tokenizer = processor = None
        with (args.output_dir / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
            for index, (key, record) in enumerate(records.items(), 1):
                if key in completed and not completed[key].get('wide_refinement_error'):
                    continue
                if key in selected:
                    if model is None:
                        model, tokenizer, processor = load_model(args.model_path,
                                                                  args.image_token_limit)
                    try:
                        result = refine_record(record, args, model, tokenizer, processor)
                    except Exception as exc:
                        result = {**record, 'wide_refinement_error':
                                  f'{type(exc).__name__}: {exc}', 'wide_refinement_steps': []}
                else:
                    result = record
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                completed[key] = result
                if key in selected:
                    steps = result.get('wide_refinement_steps') or []
                    print(f'{index}/{len(records)} {key}: splits={sum(s["accepted"] for s in steps)}, '
                          f'calls={sum(len(s["attempts"]) for s in steps)}', flush=True)
        summary = dict(total=len(records), processed=len(completed),
                       triggered=len(selected),
                       accepted=sum(any(s.get('accepted') for s in
                                        r.get('wide_refinement_steps', [])) for r in completed.values()),
                       calls=sum(len(s.get('attempts', [])) for r in completed.values()
                                 for s in r.get('wide_refinement_steps', [])),
                       errors=sum(bool(r.get('wide_refinement_error')) for r in completed.values()),
                       unchanged=sum(completed[key].get('refined_boxes') == record.get('refined_boxes')
                                     for key, record in records.items()),
                       journal=str(args.journal.resolve()),
                       denominator='all input F records; no GT accessed')
        atomic_json(args.output_dir / 'summary.json', summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return 0 if len(completed) == len(records) and summary['errors'] == 0 else 2
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
