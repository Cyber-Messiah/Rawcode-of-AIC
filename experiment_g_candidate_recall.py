#!/usr/bin/env python3
"""Recover missing same-class instances for ordinal G, without using GT.

This is a separate experiment on a saved F (or wide-refined) journal. Queries
sharing an image and target share a candidate pool. New boxes need independent
re-observation before they enter ``refined_boxes``; all calls are logged.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

from PIL import Image, ImageEnhance

from evaluate import box_iou
from experiment_f_official_multi import generate, load_model, parse_boxes
from experiment_f_recursive_multi import map_from_crop
from predict_rgb_all import atomic_json, read_progress
from prepare_reference_subset import sha256, valid_box


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def same_instance(left, right):
    """Tolerate small coordinate drift without joining adjacent objects."""
    if box_iou(left, right) >= .35:
        return True
    lx, ly = (left[0] + left[2]) / 2, (left[1] + left[3]) / 2
    rx, ry = (right[0] + right[2]) / 2, (right[1] + right[3]) / 2
    lw, lh = left[2] - left[0], left[3] - left[1]
    rw, rh = right[2] - right[0], right[3] - right[1]
    return (abs(lx - rx) <= .4 * min(lw, rw)
            and abs(ly - ry) <= .4 * min(lh, rh))


def unique_boxes(boxes):
    kept = []
    for box in boxes:
        if valid_box(box) and not any(same_instance(box, old) for old in kept):
            kept.append(list(box))
    return kept


def group_records(records):
    groups = {}
    for record in records.values():
        key = (record['visible'], ' '.join(record['target'].lower().split()))
        groups.setdefault(key, []).append(record)
    return groups


def group_seed_boxes(rows):
    return unique_boxes(box for row in rows
                        for box in (row.get('refined_boxes') or []))


def candidate_gate(rows, seed, wide_threshold=.8, area_threshold=.1,
                   probe_complete=False):
    required = max(int(row['rank']) for row in rows)
    if any(box[2] - box[0] >= wide_threshold or area(box) >= area_threshold
           for box in seed):
        return required, 'oversized_candidate_handle_class3_first'
    if len(seed) < required:
        return required, 'rank_deficit'
    if probe_complete and len(seed) <= max(required, 2):
        return required, 'probe_complete'
    return required, 'enough_candidates'


def crop_bounds(image, bounds):
    width, height = image.size
    x1 = max(0, min(width - 1, math.floor(bounds[0] * width)))
    y1 = max(0, min(height - 1, math.floor(bounds[1] * height)))
    x2 = max(x1 + 1, min(width, math.ceil(bounds[2] * width)))
    y2 = max(y1 + 1, min(height, math.ceil(bounds[3] * height)))
    return (x1, y1, x2, y2)


def search_regions(image, seed, max_gap_regions=4):
    """Use a context band and uncovered horizontal gaps, never a found-box crop."""
    if not seed:
        return []
    low = min(box[1] for box in seed)
    high = max(box[3] for box in seed)
    span = high - low
    regions = []
    if span < .55:
        y1, y2 = max(0, low - max(.05, span)), min(1, high + max(.05, span))
        if y2 - y1 < .85:
            regions.append(('context_band', crop_bounds(image, (0, y1, 1, y2))))
    intervals = sorted((box[0], box[2]) for box in seed)
    merged = []
    for left, right in intervals:
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    gaps = []
    edge = 0.0
    for left, right in merged:
        if left - edge >= .08:
            gaps.append((edge, left))
        edge = max(edge, right)
    if 1 - edge >= .08:
        gaps.append((edge, 1.0))
    for index, (left, right) in enumerate(sorted(gaps, key=lambda g: g[1] - g[0],
                                                 reverse=True)[:max_gap_regions], 1):
        # Five per mille context does not reintroduce most of the excluded box.
        bounds = crop_bounds(image, (max(0, left - .005), 0,
                                     min(1, right + .005), 1))
        regions.append((f'uncovered_region_{index}', bounds))
    return regions


def stable_new_boxes(seed, observations, min_support, min_size_ratio,
                     max_size_ratio, max_width=.8, max_area=.1):
    """Cluster boxes from distinct calls; leave weak proposals in the audit log."""
    clusters = []
    seed_areas = sorted(area(box) for box in seed)
    typical_area = seed_areas[len(seed_areas) // 2] if seed_areas else None
    for call_id, box in observations:
        if (not valid_box(box) or box[2] - box[0] >= max_width
                or area(box) >= max_area
                or any(same_instance(box, old) for old in seed)):
            continue
        if typical_area is not None and not (min_size_ratio * typical_area
                                              <= area(box) <= max_size_ratio * typical_area):
            continue
        match = next((cluster for cluster in clusters
                      if same_instance(box, cluster['box'])), None)
        if match is None:
            clusters.append(dict(box=list(box), calls={call_id}, variants=[list(box)]))
        else:
            match['calls'].add(call_id)
            match['variants'].append(list(box))
    accepted = [cluster for cluster in clusters
                if len(cluster['calls']) >= min_support]
    return unique_boxes(cluster['box'] for cluster in accepted), [
        dict(box=cluster['box'], support=len(cluster['calls'])) for cluster in clusters]


def recall_group(rows, args, model, tokenizer, processor):
    seed = group_seed_boxes(rows)
    required, reason = candidate_gate(rows, seed, args.wide_width_threshold,
                                      args.large_area_threshold, args.probe_complete)
    result = dict(required_count=required, initial_count=len(seed), gate=reason,
                  initial_boxes=seed, final_boxes=seed, accepted_new_boxes=[],
                  attempts=[], proposals=[])
    if reason not in ('rank_deficit', 'probe_complete'):
        return result
    image_path = Path(rows[0]['visible'])
    if image_path.is_absolute() or '..' in image_path.parts:
        raise ValueError(f'Unsafe image path: {image_path}')
    with Image.open(args.data_root / image_path) as source:
        image = source.convert('RGB')
    target = rows[0]['target']
    count_hint = (f'The ordinal questions require at least {required} distinct '
                  'instances if they are visible. ' if required > 1 else '')
    prompts = [
        (f'Locate every separate individual {target} visible in this image. '
         f'{count_hint}Return one tight bounding box '
         'for each individual instance, including small or less salient ones. '
         'Do not return an enclosing group box.'),
        (f'Find all distinct {target} instances, including those beside the '
         'most obvious one. Inspect the whole image from left to right. '
         'Output a separate tight box for each visible instance; do not '
         'stop after finding one and do not invent absent instances.'),
    ]
    group_id = '|'.join((rows[0]['visible'], target))
    seed_value = args.seed + int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16)
    observations = []

    def attempt(picture, bounds, prompt, kind, temperature):
        if len(result['attempts']) >= args.max_calls_per_group:
            return True
        number = len(result['attempts']) + 1
        step = dict(kind=kind, bounds=list(bounds), temperature=temperature,
                    prompt=prompt, seed=seed_value + number)
        try:
            answer = generate(model, tokenizer, processor, picture, prompt,
                              args.multi_max_new_tokens, seed_value + number,
                              temperature, args.image_token_limit)
            local = parse_boxes(answer)
            mapped = [map_from_crop(box, bounds, image.size) for box in local]
            mapped = unique_boxes(mapped)
            observations.extend((number, box) for box in mapped)
            step.update(answer=answer, boxes=mapped)
        except Exception as exc:
            step['error'] = f'{type(exc).__name__}: {exc}'
        result['attempts'].append(step)
        accepted, proposals = stable_new_boxes(
            seed, observations, args.min_support, args.min_size_ratio,
            args.max_size_ratio, args.wide_width_threshold,
            args.large_area_threshold)
        result['proposals'] = proposals
        result['accepted_new_boxes'] = accepted
        result['final_boxes'] = unique_boxes(seed + accepted)
        return (len(result['final_boxes']) >= required
                and reason != 'probe_complete')

    full = (0, 0, image.width, image.height)
    for index, prompt in enumerate(prompts, 1):
        if attempt(image, full, prompt, f'full_prompt_{index}', args.temperature):
            return result
    for name, bounds in search_regions(image, seed, args.max_gap_regions):
        crop = image.crop(bounds)
        for index, prompt in enumerate(prompts, 1):
            if attempt(crop, bounds, prompt, f'{name}_prompt_{index}',
                       args.temperature):
                return result
    for factor in (1 + args.brightness_delta, 1 - args.brightness_delta):
        changed = ImageEnhance.Brightness(image).enhance(factor)
        if attempt(changed, full, prompts[0], f'brightness_{factor:.3f}',
                   args.temperature):
            return result
    attempt(image, full, prompts[1], 'temperature_retry',
            min(1.0, args.temperature + args.temperature_delta))
    return result


def apply_recall(row, recalled):
    if recalled is None:
        return row
    if recalled['accepted_new_boxes']:
        return dict(row, refined_boxes=unique_boxes(
            (row.get('refined_boxes') or []) + recalled['final_boxes']),
            recall=recalled)
    return dict(row, recall=recalled)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=root / 'datasets/reference_subset')
    parser.add_argument('--model-path', type=Path, default=root / 'LocateAnything-3B')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--ids', nargs='*', help='Select groups containing these IDs; updates all sibling IDs')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--temperature', type=float, default=.4)
    parser.add_argument('--temperature-delta', type=float, default=.15)
    parser.add_argument('--brightness-delta', type=float, default=.08)
    parser.add_argument('--image-token-limit', type=int, default=4096)
    parser.add_argument('--multi-max-new-tokens', type=int, default=2048)
    parser.add_argument('--max-calls-per-group', type=int, default=12)
    parser.add_argument('--max-gap-regions', type=int, default=4)
    parser.add_argument('--min-support', type=int, default=2)
    parser.add_argument('--min-size-ratio', type=float, default=.15)
    parser.add_argument('--max-size-ratio', type=float, default=6)
    parser.add_argument('--wide-width-threshold', type=float, default=.8)
    parser.add_argument('--large-area-threshold', type=float, default=.1)
    parser.add_argument('--probe-complete', action='store_true',
                        help='Also re-query groups with only just-enough candidates')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if (args.seed < 0 or not 0 <= args.temperature <= 1
            or not 0 <= args.temperature_delta <= 1
            or not 0 <= args.brightness_delta <= .25
            or args.image_token_limit < 1 or args.multi_max_new_tokens < 1
            or args.max_calls_per_group < 1 or args.max_gap_regions < 0
            or args.min_support < 1 or not 0 < args.min_size_ratio <= 1
            or args.max_size_ratio < 1 or not 0 < args.wide_width_threshold <= 1
            or not 0 < args.large_area_threshold <= 1):
        parser.error('Invalid numeric arguments')
    records = read_progress(args.journal)
    if not records:
        raise ValueError('Empty F journal')
    if args.ids is not None and set(args.ids) - set(records):
        raise ValueError(f'Unknown IDs: {sorted(set(args.ids) - set(records))}')
    groups = group_records(records)
    selected = {key: rows for key, rows in groups.items()
                if args.ids is None or any(row['id'] in args.ids for row in rows)}
    audit = []
    for key, rows in selected.items():
        seed = group_seed_boxes(rows)
        required, reason = candidate_gate(rows, seed, args.wide_width_threshold,
                                          args.large_area_threshold, args.probe_complete)
        audit.append(dict(image=key[0], target=key[1], ids=[r['id'] for r in rows],
                          required=required, candidates=len(seed), gate=reason))
    counts = {reason: sum(row['gate'] == reason for row in audit)
              for reason in sorted({row['gate'] for row in audit})}
    print(json.dumps(dict(total_records=len(records), selected_groups=len(selected),
                          gate_counts=counts,
                          triggered_groups=[row for row in audit if row['gate'] in
                                            ('rank_deficit', 'probe_complete')]),
                     ensure_ascii=False, indent=2),
          flush=True)
    if args.check_only:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = dict(script_sha256=sha256(Path(__file__)),
                  journal_sha256=sha256(args.journal), ids=args.ids,
                  data_root=str(args.data_root.resolve()),
                  model_path=str(args.model_path.resolve()),
                  settings={name: getattr(args, name) for name in (
                      'seed', 'temperature', 'temperature_delta', 'brightness_delta',
                      'image_token_limit', 'multi_max_new_tokens', 'max_calls_per_group',
                      'max_gap_regions', 'min_support', 'min_size_ratio', 'max_size_ratio',
                      'wide_width_threshold', 'large_area_threshold', 'probe_complete')})
    manifest = args.output_dir / 'run_config.json'
    if manifest.exists():
        if json.loads(manifest.read_text(encoding='utf-8')) != config:
            raise ValueError('Run configuration changed; choose a new output directory')
    else:
        if (args.output_dir / 'predictions.jsonl').exists():
            raise ValueError('Existing journal without run_config.json')
        atomic_json(manifest, config)
    atomic_json(args.output_dir / 'gate_audit.json', audit)
    lock = args.output_dir / 'run.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        completed = read_progress(args.output_dir / 'predictions.jsonl')
        model = tokenizer = processor = None
        selected_ids = {row['id'] for rows in selected.values() for row in rows}
        with (args.output_dir / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
            for key, rows in groups.items():
                if all(row['id'] in completed for row in rows):
                    continue
                if key in selected:
                    seed = group_seed_boxes(rows)
                    _, reason = candidate_gate(rows, seed, args.wide_width_threshold,
                                               args.large_area_threshold, args.probe_complete)
                    if reason in ('rank_deficit', 'probe_complete'):
                        if model is None:
                            model, tokenizer, processor = load_model(
                                args.model_path, args.image_token_limit)
                        try:
                            recalled = recall_group(rows, args, model, tokenizer, processor)
                        except Exception as exc:
                            recalled = dict(gate=reason, initial_boxes=seed,
                                            final_boxes=seed, accepted_new_boxes=[],
                                            attempts=[], proposals=[],
                                            error=f'{type(exc).__name__}: {exc}')
                    else:
                        recalled = dict(gate=reason, initial_boxes=seed,
                                        final_boxes=seed, accepted_new_boxes=[],
                                        attempts=[], proposals=[])
                else:
                    recalled = None
                for row in rows:
                    result = apply_recall(row, recalled)
                    stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    completed[row['id']] = result
                if recalled is not None:
                    print(f'{key[0]} | {key[1]}: {recalled["gate"]}, '
                          f'{len(recalled["initial_boxes"])} -> '
                          f'{len(recalled["final_boxes"])} candidates, '
                          f'{len(recalled["attempts"])} calls', flush=True)
        summary = dict(total_records=len(records), processed=len(completed),
                       selected_records=len(selected_ids), selected_groups=len(selected),
                       gate_counts=counts,
                       groups_with_new_candidates=sum(
                           len(completed[rows[0]['id']].get('recall', {}).get('final_boxes', []))
                           > len(completed[rows[0]['id']].get('recall', {}).get('initial_boxes', []))
                           for rows in selected.values()),
                       calls=sum(len(completed[rows[0]['id']].get('recall', {}).get('attempts', []))
                                 for rows in selected.values()),
                       errors=sum(bool(completed[rows[0]['id']].get('recall', {}).get('error'))
                                  for rows in selected.values()),
                       denominator='all input F records; no GT accessed')
        atomic_json(args.output_dir / 'summary.json', summary)
        return 0 if len(completed) == len(records) and summary['errors'] == 0 else 2
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
