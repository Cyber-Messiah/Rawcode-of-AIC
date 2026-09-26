#!/usr/bin/env python3
"""Full final-round RGB inference: A for general queries, G-prime for clear ordinals.

G-prime mirrors the preliminary wide-split and candidate-recall experiments.
No reference boxes are read. All output boxes are normalized xyxy.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

from bbox_coordinates import parse_boxes
from predict_bbox import atomic_json, find_queries, read_progress, valid_box
import predict_bbox_v2 as base


def width(box):
    return box[2] - box[0]


def same_instance(left, right):
    if base.iou(left, right) >= .35:
        return True
    lx, ly = (left[0] + left[2]) / 2, (left[1] + left[3]) / 2
    rx, ry = (right[0] + right[2]) / 2, (right[1] + right[3]) / 2
    return (abs(lx - rx) <= .4 * min(width(left), width(right))
            and abs(ly - ry) <= .4 * min(left[3] - left[1], right[3] - right[1]))


def unique_boxes(boxes):
    kept = []
    for box in boxes:
        if valid_box(box) and not any(same_instance(box, old) for old in kept):
            kept.append(list(box))
    return kept


def replace_parent(active, parent, children, threshold=.4):
    position = next((i for i, box in enumerate(active) if base.iou(box, parent) >= .999), None)
    if position is None:
        return base.deduplicate(active + children, threshold)
    return base.deduplicate(active[:position] + children + active[position + 1:], threshold)


def horizontal_tiles(image, parent, fraction=.42, overlap=.15, padding=.02):
    _, (left, top, right, bottom) = base.crop_region(image, parent, padding)
    span = right - left
    tile_width = max(1, min(span, math.ceil(span * fraction)))
    if tile_width == span:
        return [(image.crop((left, top, right, bottom)), (left, top, right, bottom))]
    stride = max(1, round(tile_width * (1 - overlap)))
    starts = list(range(left, right - tile_width + 1, stride))
    if not starts or starts[-1] != right - tile_width:
        starts.append(right - tile_width)
    return [(image.crop((x, top, x + tile_width, bottom)),
             (x, top, x + tile_width, bottom)) for x in starts]


def strict_children(local_boxes, bounds, image_size, parent, local_width_limit=1):
    accepted = []
    for local in local_boxes:
        if width(local) >= local_width_limit:
            continue
        child = base.map_from_crop(local, bounds, image_size)
        if (valid_box(child) and width(child) <= width(parent) * .6
                and base.area(child) <= base.area(parent) * .45
                and base.intersection(child, parent) / base.area(child) >= .9):
            accepted.append(child)
    return base.deduplicate(accepted, .4)


def split_wide_parent(image, active, parent, rank, target, seed, generate, args):
    crop, bounds = base.crop_region(image, parent, .02)
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

    def attempt(picture, picture_bounds, prompt, kind, local_limit=1):
        step = dict(kind=kind, prompt=prompt, bounds=list(picture_bounds))
        try:
            answer = generate(picture, prompt, seed + 1000 + len(attempts) + 1,
                              args.multi_max_new_tokens, args.multi_image_token_limit,
                              args.gprime_temperature)
            raw, audit = parse_boxes(answer, picture.size)
            children = strict_children(raw, picture_bounds, image.size, parent, local_limit)
            collected[:] = base.deduplicate(collected + children, .4)
            step.update(answer=answer, audit=audit, accepted_children=children)
        except Exception as exc:
            step['error'] = f'{type(exc).__name__}: {exc}'
        attempts.append(step)
        replaced = replace_parent(active, parent, collected)
        return (len(collected) >= 2 and len(replaced) >= rank
                and len(replaced) > len(active))

    for index, prompt in enumerate(prompts, 1):
        if attempt(crop, bounds, prompt, f'full_crop_prompt_{index}'):
            return collected, attempts
    tile_prompt = (f'Locate each separate {target} in this cropped image. '
                   'Return one tight box per individual object. '
                   'Ignore objects not visible in this crop; never box the whole crop.')
    for index, (tile, tile_bounds) in enumerate(horizontal_tiles(image, parent), 1):
        if attempt(tile, tile_bounds, tile_prompt, f'horizontal_tile_{index}', .85):
            return collected, attempts
    return [], attempts


def refine_wide(record, image, generate, args):
    active = [list(box) for box in record.get('candidate_boxes', []) if valid_box(box)]
    pending = sorted((box for box in active if width(box) >= .8),
                     key=width, reverse=True)
    steps = []
    seed = args.seed + int(hashlib.sha256(record['id'].encode()).hexdigest()[:8], 16)
    for index, parent in enumerate(pending):
        if not any(base.iou(parent, box) >= .999 for box in active):
            continue
        children, attempts = split_wide_parent(
            image, active, parent, record['rank'], record['target'],
            seed + index * 100, generate, args)
        if children:
            active = replace_parent(active, parent, children)
        steps.append(dict(parent=parent, parent_width=width(parent),
                          accepted=bool(children), children=children, attempts=attempts))
    return active, steps


def candidate_gate(rows, seed, probe_complete=False):
    required = max(row['rank'] for row in rows)
    if any(width(box) >= .8 or base.area(box) >= .1 for box in seed):
        return required, 'oversized_candidate_handle_class3_first'
    if len(seed) < required:
        return required, 'rank_deficit'
    if probe_complete and len(seed) <= max(required, 2):
        return required, 'probe_complete'
    return required, 'enough_candidates'


def crop_bounds(image, bounds):
    w, h = image.size
    x1 = max(0, min(w - 1, math.floor(bounds[0] * w)))
    y1 = max(0, min(h - 1, math.floor(bounds[1] * h)))
    x2 = max(x1 + 1, min(w, math.ceil(bounds[2] * w)))
    y2 = max(y1 + 1, min(h, math.ceil(bounds[3] * h)))
    return (x1, y1, x2, y2)


def search_regions(image, seed, max_gap_regions=4):
    if not seed:
        return []
    low, high = min(box[1] for box in seed), max(box[3] for box in seed)
    span = high - low
    regions = []
    if span < .55:
        y1, y2 = max(0, low - max(.05, span)), min(1, high + max(.05, span))
        if y2 - y1 < .85:
            regions.append(('context_band', crop_bounds(image, (0, y1, 1, y2))))
    merged = []
    for left, right in sorted((box[0], box[2]) for box in seed):
        if merged and left <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], right))
        else:
            merged.append((left, right))
    gaps, edge = [], 0.0
    for left, right in merged:
        if left - edge >= .08:
            gaps.append((edge, left))
        edge = max(edge, right)
    if 1 - edge >= .08:
        gaps.append((edge, 1.0))
    for index, (left, right) in enumerate(sorted(gaps, key=lambda gap: gap[1] - gap[0],
                                                 reverse=True)[:max_gap_regions], 1):
        regions.append((f'uncovered_region_{index}', crop_bounds(
            image, (max(0, left - .005), 0, min(1, right + .005), 1))))
    return regions


def stable_new_boxes(seed, observations, min_support=2):
    clusters = []
    areas = sorted(base.area(box) for box in seed)
    typical = areas[len(areas) // 2] if areas else None
    for call_id, box in observations:
        if (not valid_box(box) or width(box) >= .8 or base.area(box) >= .1
                or any(same_instance(box, old) for old in seed)):
            continue
        if typical is not None and not (.15 * typical <= base.area(box) <= 6 * typical):
            continue
        match = next((cluster for cluster in clusters
                      if same_instance(box, cluster['box'])), None)
        if match is None:
            clusters.append(dict(box=list(box), calls={call_id}))
        else:
            match['calls'].add(call_id)
    accepted = [cluster for cluster in clusters if len(cluster['calls']) >= min_support]
    return unique_boxes(cluster['box'] for cluster in accepted), [
        dict(box=cluster['box'], support=len(cluster['calls'])) for cluster in clusters]


def recall_group(rows, image, generate, args):
    from PIL import ImageEnhance

    seed = unique_boxes(box for row in rows for box in row['wide_boxes'])
    required, reason = candidate_gate(rows, seed, args.probe_complete)
    result = dict(required_count=required, initial_count=len(seed), gate=reason,
                  initial_boxes=seed, final_boxes=seed, accepted_new_boxes=[],
                  attempts=[], proposals=[])
    if reason not in ('rank_deficit', 'probe_complete'):
        return result
    target = rows[0]['target']
    count_hint = (f'The ordinal questions require at least {required} distinct '
                  'instances if they are visible. ' if required > 1 else '')
    prompts = [
        (f'Locate every separate individual {target} visible in this image. '
         f'{count_hint}Return one tight bounding box for each individual instance, '
         'including small or less salient ones. Do not return an enclosing group box.'),
        (f'Find all distinct {target} instances, including those beside the most '
         'obvious one. Inspect the whole image from left to right. Output a separate '
         'tight box for each visible instance; do not stop after finding one and '
         'do not invent absent instances.'),
    ]
    group_id = '|'.join((rows[0]['visible'], target))
    seed_value = args.seed + int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16)
    observations = []

    def attempt(picture, bounds, prompt, kind, temperature):
        if len(result['attempts']) >= args.max_recall_calls:
            return True
        number = len(result['attempts']) + 1
        step = dict(kind=kind, bounds=list(bounds), temperature=temperature,
                    prompt=prompt, seed=seed_value + number)
        try:
            answer = generate(picture, prompt, seed_value + number,
                              args.multi_max_new_tokens, args.multi_image_token_limit,
                              temperature)
            local, audit = parse_boxes(answer, picture.size)
            mapped = unique_boxes(base.map_from_crop(box, bounds, image.size)
                                  for box in local)
            observations.extend((number, box) for box in mapped)
            step.update(answer=answer, audit=audit, boxes=mapped)
        except Exception as exc:
            step['error'] = f'{type(exc).__name__}: {exc}'
        result['attempts'].append(step)
        accepted, proposals = stable_new_boxes(seed, observations)
        result['proposals'] = proposals
        result['accepted_new_boxes'] = accepted
        result['final_boxes'] = unique_boxes(seed + accepted)
        return len(result['final_boxes']) >= required and reason != 'probe_complete'

    full = (0, 0, image.width, image.height)
    for index, prompt in enumerate(prompts, 1):
        if attempt(image, full, prompt, f'full_prompt_{index}', args.gprime_temperature):
            return result
    for name, bounds in search_regions(image, seed):
        crop = image.crop(bounds)
        for index, prompt in enumerate(prompts, 1):
            if attempt(crop, bounds, prompt, f'{name}_prompt_{index}',
                       args.gprime_temperature):
                return result
    for factor in (1 + args.brightness_delta, 1 - args.brightness_delta):
        changed = ImageEnhance.Brightness(image).enhance(factor)
        if attempt(changed, full, prompts[0], f'brightness_{factor:.3f}',
                   args.gprime_temperature):
            return result
    attempt(image, full, prompts[1], 'temperature_retry',
            min(1.0, args.gprime_temperature + args.temperature_delta))
    return result


def select_final(row, boxes, wide_steps, recall):
    result = dict(row)
    result.update(wide_boxes=row.get('wide_boxes', row.get('candidate_boxes', [])),
                  wide_refinement_steps=wide_steps, recall=recall,
                  gprime_boxes=boxes)
    if not row.get('ordinal'):
        return result
    selected = base.select_g(boxes, row['rank'], row['direction'])
    if selected is not None:
        source = ('gprime_recall' if recall['accepted_new_boxes'] else
                  'gprime_wide' if any(step['accepted'] for step in wide_steps) else
                  row.get('source') if str(row.get('source', '')).startswith('g_') else
                  'gprime_initial')
        result.update(status='ok', bbox=selected, source=source)
    elif valid_box(row.get('a_bbox')):
        result.update(status='ok', bbox=row['a_bbox'], source='a_ordinal_fallback')
    else:
        result.update(status='error', bbox=None, source='none',
                      gprime_error='insufficient_candidates_and_no_a_box')
    return result


def selected_rows(args):
    queries_path = find_queries(args.data_root, args.queries)
    rows = base.load_queries(queries_path, args.data_root, not args.skip_image_check)
    available = len(rows)
    if args.only_ordinals:
        rows = [(key, item) for key, item in rows if base.parse_ordinal(item['query'])]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError('No selected queries')
    return queries_path, rows, available


def group_ordinals(rows, initial):
    groups = {}
    for key, item in rows:
        ordinal = base.parse_ordinal(item['query'])
        if ordinal:
            record = dict(initial[key], id=key, visible=item['visible'],
                          rank=ordinal[0], direction=ordinal[1], target=ordinal[2],
                          ordinal=True)
            group_key = (item['visible'], ' '.join(ordinal[2].lower().split()))
            groups.setdefault(group_key, []).append(record)
    return groups


def export(rows, records, output, fallback):
    summary = base.export(rows, records, output, fallback)
    sources = Counter(record.get('source', 'none') for record in records.values())
    summary.update(strategy='A + G-prime (preliminary wide split and recall)',
                   sources=dict(sources),
                   wide_triggered=sum(bool(record.get('wide_refinement_steps'))
                                      for record in records.values()),
                   wide_accepted=sum(any(step['accepted'] for step in
                                           record.get('wide_refinement_steps', []))
                                     for record in records.values()),
                   recall_triggered_groups=len({(record.get('visible'), record.get('target'))
                       for record in records.values() if record.get('recall', {}).get('gate')
                       in ('rank_deficit', 'probe_complete')}),
                   note='No GT used; finals accuracy requires reference boxes')
    atomic_json(output / 'summary.json', summary)
    return summary


def successful(record):
    return bool(record and record.get('status') == 'ok' and
                valid_box(record.get('bbox')))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--queries', type=Path)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--only-ordinals', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image-token-limit', type=int, default=25600)
    parser.add_argument('--multi-image-token-limit', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--multi-max-new-tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=.7)
    parser.add_argument('--gprime-temperature', type=float, default=.4)
    parser.add_argument('--temperature-delta', type=float, default=.15)
    parser.add_argument('--brightness-delta', type=float, default=.08)
    parser.add_argument('--top-p', type=float, default=.9)
    parser.add_argument('--retries', type=int, default=1)
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--max-recall-calls', type=int, default=12)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--probe-complete', action='store_true')
    parser.add_argument('--fallback-full-image', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--skip-image-check', action='store_true')
    args = parser.parse_args(argv)
    if (args.limit < 0 or args.seed < 0 or args.retries < 0 or args.max_depth < 0
            or min(args.image_token_limit, args.multi_image_token_limit,
                   args.max_new_tokens, args.multi_max_new_tokens,
                   args.max_recall_calls, args.save_every) < 1
            or not 0 < args.temperature <= 1 or not 0 < args.gprime_temperature <= 1
            or not 0 <= args.temperature_delta <= 1
            or not 0 <= args.brightness_delta <= .25 or not 0 < args.top_p <= 1):
        parser.error('invalid numeric settings')
    if args.skip_image_check and not args.check_only:
        parser.error('--skip-image-check is only valid with --check-only')
    args.data_root = args.data_root.resolve()
    args.model_path = args.model_path.resolve()
    queries_path, rows, available = selected_rows(args)
    ordinal_count = sum(bool(base.parse_ordinal(item['query'])) for _, item in rows)
    print(f'Validated {len(rows)}/{available} queries; {ordinal_count} routed to G-prime',
          flush=True)
    if args.check_only:
        return 0
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'run.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        settings = {key: value for key, value in vars(args).items()
                    if key not in ('output_dir', 'fallback_full_image', 'save_every')}
        settings = {key: str(value) if isinstance(value, Path) else value
                    for key, value in settings.items()}
        config = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      base_sha256=hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest(),
                      queries_sha256=hashlib.sha256(queries_path.read_bytes()).hexdigest(),
                      selected_ids=[key for key, _ in rows], settings=settings)
        manifest = output / 'run_config.json'
        if manifest.exists():
            if json.loads(manifest.read_text(encoding='utf-8')) != config:
                raise ValueError('Run configuration changed; choose a new --output-dir')
        else:
            if (output / 'predictions.jsonl').exists():
                raise ValueError('Existing journal without run_config.json')
            atomic_json(manifest, config)

        records = read_progress(output / 'predictions.jsonl')
        if all(successful(records.get(key)) for key, _ in rows):
            summary = export(rows, records, output, args.fallback_full_image)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
            return 0 if summary['submission_ready'] else 2

        initial_dir = output / '_initial_g'
        initial_args = ['--data-root', str(args.data_root), '--queries', str(queries_path),
                        '--model-path', str(args.model_path), '--output-dir', str(initial_dir),
                        '--seed', str(args.seed), '--image-token-limit', str(args.image_token_limit),
                        '--multi-image-token-limit', str(args.multi_image_token_limit),
                        '--max-new-tokens', str(args.max_new_tokens),
                        '--multi-max-new-tokens', str(args.multi_max_new_tokens),
                        '--temperature', str(args.temperature), '--top-p', str(args.top_p),
                        '--retries', str(args.retries), '--max-depth', str(args.max_depth),
                        '--save-every', str(args.save_every)]
        if args.limit:
            initial_args += ['--limit', str(args.limit)]
        if args.only_ordinals:
            initial_args.append('--only-ordinals')
        base.main(initial_args)
        initial = read_progress(initial_dir / 'predictions.jsonl')
        missing = [key for key, _ in rows if key not in initial]
        if missing:
            raise RuntimeError(f'Initial G stage incomplete: {len(missing)} missing records')
        groups = group_ordinals(rows, initial)
        ordinal_ids = {row['id'] for group in groups.values() for row in group}
        generate = None

        def get_generator():
            nonlocal generate
            if generate is None:
                generate = base.make_generator(args)
            return generate

        started = time.monotonic()
        with (output / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
            def save(row):
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                records[row['id']] = row

            for key, item in rows:
                if key not in ordinal_ids and not successful(records.get(key)):
                    save(initial[key])
            for index, (group_key, members) in enumerate(groups.items(), 1):
                if all(successful(records.get(row['id'])) for row in members):
                    continue
                from PIL import Image
                with Image.open(args.data_root / group_key[0]) as opened:
                    image = opened.convert('RGB')
                prepared = []
                for row in members:
                    try:
                        boxes = row.get('candidate_boxes', [])
                        if any(width(box) >= .8 for box in boxes):
                            boxes, steps = refine_wide(row, image, get_generator(), args)
                        else:
                            steps = []
                        prepared.append(dict(row, wide_boxes=boxes,
                                             wide_refinement_steps=steps))
                    except Exception as exc:
                        prepared.append(dict(row, wide_boxes=row.get('candidate_boxes', []),
                                             wide_refinement_steps=[],
                                             wide_error=f'{type(exc).__name__}: {exc}'))
                try:
                    seed_boxes = unique_boxes(box for row in prepared
                                              for box in row['wide_boxes'])
                    _, gate = candidate_gate(prepared, seed_boxes, args.probe_complete)
                    recall = recall_group(prepared, image,
                                          get_generator() if gate in
                                          ('rank_deficit', 'probe_complete') else None, args)
                except Exception as exc:
                    recall = dict(gate='error', initial_boxes=[], final_boxes=[],
                                  accepted_new_boxes=[], attempts=[], proposals=[],
                                  error=f'{type(exc).__name__}: {exc}')
                for row in prepared:
                    boxes = unique_boxes(row['wide_boxes'] +
                                         (recall['final_boxes'] if
                                          recall['accepted_new_boxes'] else []))
                    final = select_final(row, boxes,
                                         row['wide_refinement_steps'], recall)
                    save(final)
                if index % args.save_every == 0 or recall.get('attempts'):
                    print(f'G-prime groups {index}/{len(groups)}; '
                          f'gate={recall["gate"]}; candidates='
                          f'{len(recall["final_boxes"])}; '
                          f'elapsed={time.monotonic() - started:.0f}s', flush=True)
                if index % args.save_every == 0:
                    export(rows, records, output, args.fallback_full_image)
        summary = export(rows, records, output, args.fallback_full_image)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0 if summary['submission_ready'] else 2
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
