#!/usr/bin/env python3
"""Final-round RGB prediction: A for general queries, refined G for clear ordinals.

All exported boxes are normalized xyxy. The parser treats generated coordinates
as two corners and canonicalizes their order before checking geometry.
"""
import argparse
from collections import Counter
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time

from predict_bbox import atomic_json, find_queries, read_progress, valid_box


BOX = re.compile(r'<box>(.*?)</box>', re.I | re.S)
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
TAGGED = re.compile(r'\s*' + r'\s*'.join(fr'<({NUMBER})>' for _ in range(4)) + r'\s*')
PLAIN = re.compile(r'\s*[\[(]?\s*' + r'\s*,\s*'.join(fr'({NUMBER})' for _ in range(4)) + r'\s*[\])]??\s*')
DIRECTION = re.compile(r'\b(left\s+to\s+right|right\s+to\s+left)\b', re.I)
ORDER_PHRASE = re.compile(r'\b(?:from\s+(?:the\s+)?)?(?:left\s+to\s+right|right\s+to\s+left)\b', re.I)
VERTICAL = re.compile(r'\b(?:top\s+to\s+bottom|bottom\s+to\s+top)\b', re.I)
ORDINAL = re.compile(r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|[1-9](?:st|nd|rd|th)|10th)\b', re.I)
WORDS = {word: number for number, word in enumerate(
    ('first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth'), 1)}


def parse_ordinal(query):
    directions, ranks = list(DIRECTION.finditer(query)), list(ORDINAL.finditer(query))
    if len(directions) != 1 or len(ranks) != 1 or VERTICAL.search(query):
        return None
    token = ranks[0].group().lower()
    rank = WORDS[token] if token in WORDS else int(re.match(r'\d+', token).group())
    direction = 'left_to_right' if directions[0].group().lower().startswith('left') else 'right_to_left'
    target = ORDER_PHRASE.sub(' ', query)
    target = ORDINAL.sub(' ', target)
    target = re.sub(r'\bfrom\s+the\s+(?:left|right)\s*,?\s*when\s+counting\b', ' ', target, flags=re.I)
    target = re.sub(r'[(),;:.]+', ' ', target)
    target = re.sub(r'\s+', ' ', target).strip()
    target = re.sub(r'^(?:count|number)\s+(?:the\s+)?', '', target, flags=re.I).strip()
    target = re.sub(r'^(?:(?:from|the|a|an)\s+)+', '', target, flags=re.I).strip()
    return (rank, direction, target) if target else None


def parse_boxes(answer, image_size=None):
    """Return valid boxes and a parse audit; never silently clip or double-scale.

    Tagged coordinates are LocateAnything's 0..1000 units. Plain coordinates
    in 0..1 are normalized; otherwise 0..1000 are model units. Values above
    1000 can only be interpreted as pixels when image dimensions permit it.
    """
    boxes, audit = [], dict(box_tags=0, accepted=0, reversed_corners=0,
                            units=Counter(), rejected=Counter(), explicit_none=0)
    for match in BOX.finditer(str(answer)):
        audit['box_tags'] += 1
        body = match.group(1).strip()
        if body.lower() in ('none', 'null'):
            audit['explicit_none'] += 1
            continue
        tagged = TAGGED.fullmatch(body)
        found = tagged or PLAIN.fullmatch(body)
        if not found:
            audit['rejected']['syntax'] += 1
            continue
        numbers = [float(v) for v in found.groups()]
        if any(not math.isfinite(v) or v < 0 for v in numbers):
            audit['rejected']['nonfinite_or_negative'] += 1
            continue
        if tagged:
            unit = 'model_1000'
            if any(v > 1000 for v in numbers):
                audit['rejected']['tagged_out_of_range'] += 1
                continue
        elif all(v <= 1 for v in numbers):
            unit = 'normalized'
        elif all(v <= 1000 for v in numbers):
            unit = 'model_1000'
        elif (image_size and image_size[0] > 0 and image_size[1] > 0
              and numbers[0] <= image_size[0] and numbers[2] <= image_size[0]
              and numbers[1] <= image_size[1] and numbers[3] <= image_size[1]):
            unit = 'pixel'
        else:
            audit['rejected']['out_of_range_or_ambiguous_pixels'] += 1
            continue
        if unit == 'model_1000':
            values = [v / 1000.0 for v in numbers]
        elif unit == 'pixel':
            values = [numbers[i] / image_size[i % 2] for i in range(4)]
        else:
            values = numbers
        reversed_corners = values[0] > values[2] or values[1] > values[3]
        box = [min(values[0], values[2]), min(values[1], values[3]),
               max(values[0], values[2]), max(values[1], values[3])]
        if not valid_box(box):
            audit['rejected']['degenerate_or_out_of_range'] += 1
            continue
        boxes.append(box)
        audit['accepted'] += 1
        audit['reversed_corners'] += int(reversed_corners)
        audit['units'][unit] += 1
    audit['units'], audit['rejected'] = dict(audit['units']), dict(audit['rejected'])
    return boxes, audit


def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    overlap = intersection(a, b)
    return overlap / (area(a) + area(b) - overlap) if overlap else 0.0


def deduplicate(boxes, threshold=0.5):
    kept = []
    for box in boxes:
        if all(iou(box, prior) <= threshold for prior in kept):
            kept.append(box)
    return kept


def remove_parents(boxes):
    kept = []
    for index, outer in enumerate(boxes):
        children = []
        for other_index, inner in enumerate(boxes):
            if index == other_index or area(outer) < 1.5 * area(inner):
                continue
            if intersection(outer, inner) / area(inner) < 0.9:
                continue
            if all(iou(inner, previous) <= 0.5 for previous in children):
                children.append(inner)
        if len(children) < 2:
            kept.append(outer)
    return kept


def suspicious_parent(boxes, baseline=None):
    if not boxes:
        return None, []
    order = sorted(range(len(boxes)), key=lambda i: area(boxes[i]), reverse=True)
    largest = boxes[order[0]]
    reasons = []
    if len(boxes) == 1 and area(largest) >= 0.10:
        reasons.append('single_large_box')
    elif len(boxes) > 1 and area(largest) >= 0.10 and area(largest) / area(boxes[order[1]]) >= 1.75:
        reasons.append('relative_area_outlier')
    if (valid_box(baseline) and area(largest) >= 0.02
            and intersection(largest, baseline) / area(baseline) >= 0.9
            and area(largest) / area(baseline) >= 3.0):
        reasons.append('encloses_a_box')
    return (order[0], reasons) if reasons else (None, [])


def crop_region(image, box, padding=0.02):
    w, h = image.size
    dx, dy = (box[2] - box[0]) * padding, (box[3] - box[1]) * padding
    bounds = (max(0, min(w - 1, math.floor((box[0] - dx) * w))),
              max(0, min(h - 1, math.floor((box[1] - dy) * h))),
              max(1, min(w, math.ceil((box[2] + dx) * w))),
              max(1, min(h, math.ceil((box[3] + dy) * h))))
    left, top, right, bottom = bounds
    bounds = (left, top, max(left + 1, right), max(top + 1, bottom))
    return image.crop(bounds), bounds


def map_from_crop(box, bounds, image_size):
    left, top, right, bottom = bounds
    w, h = image_size
    mapped = [(left + box[0] * (right - left)) / w,
              (top + box[1] * (bottom - top)) / h,
              (left + box[2] * (right - left)) / w,
              (top + box[3] * (bottom - top)) / h]
    return mapped if valid_box(mapped) else None


def select_g(boxes, rank, direction):
    ordered = sorted(boxes, key=lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2))
    if len(ordered) < rank:
        return None
    return ordered[rank - 1] if direction == 'left_to_right' else ordered[-rank]


def answer_text(answer, tokenizer):
    if isinstance(answer, str):
        return answer
    if isinstance(answer, (tuple, list)) and answer:
        return answer_text(answer[0], tokenizer)
    return tokenizer.batch_decode(answer, skip_special_tokens=False)[0]


def make_generator(args):
    import numpy as np
    import torch
    from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --check-only for input and routing validation')
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    config._attn_implementation = 'sdpa'
    config.text_config._attn_implementation = 'sdpa'
    config.text_config._attn_implementation_internal = 'sdpa'
    config.vision_config._attn_implementation = 'sdpa'
    model = AutoModel.from_pretrained(args.model_path, config=config,
                                     torch_dtype=torch.bfloat16, trust_remote_code=True)
    model.language_model.model._attn_implementation = 'sdpa'
    model = model.to('cuda').eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer = tokenizer

    def generate(image, prompt, seed, max_tokens, token_limit):
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        processor.image_processor.in_token_limit = token_limit
        messages = [{'role': 'user', 'content': [
            {'type': 'image', 'image': image}, {'type': 'text', 'text': prompt}]}]
        formatted = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos = processor.process_vision_info(messages)
        inputs = processor(text=[formatted], images=images, videos=videos, return_tensors='pt')
        generated = None
        try:
            with torch.inference_mode():
                generated = model.generate(
                    pixel_values=inputs['pixel_values'].to(device='cuda', dtype=torch.bfloat16),
                    input_ids=inputs['input_ids'].cuda(),
                    attention_mask=inputs['attention_mask'].cuda(),
                    image_grid_hws=torch.as_tensor(inputs['image_grid_hws'], device='cuda'),
                    tokenizer=tokenizer, use_cache=True, max_new_tokens=max_tokens,
                    generation_mode='hybrid', do_sample=True,
                    temperature=args.temperature, top_p=args.top_p, repetition_penalty=1.1)
            return answer_text(generated, tokenizer)
        finally:
            del inputs, generated
            gc.collect()
            torch.cuda.empty_cache()

    return generate


def infer(index, key, item, args, generate):
    from PIL import Image

    with Image.open(args.data_root / item['visible']) as opened:
        image = opened.convert('RGB')
    ordinal = parse_ordinal(item['query'])
    result = dict(status='error', bbox=None, source='none', ordinal=bool(ordinal),
                  a_answer='', a_audit={}, a_errors=[])
    a_prompt = 'Locate a single instance that matches the following description: ' + item['query']
    a_token_limit = args.image_token_limit
    for attempt in range(args.retries + 1):
        try:
            answer = generate(image, a_prompt, args.seed + index * 1009 + attempt,
                              args.max_new_tokens, a_token_limit)
            boxes, audit = parse_boxes(answer, image.size)
            result.update(a_answer=answer, a_audit=audit)
            if boxes:
                result.update(status='ok', bbox=boxes[0], source='a', a_attempts=attempt + 1)
                break
            result['a_errors'].append('no_valid_bbox')
        except Exception as exc:
            result['a_errors'].append(f'{type(exc).__name__}: {exc}')
            if 'out of memory' in str(exc).lower():
                a_token_limit = max(256, a_token_limit // 2)
    if not ordinal:
        return result
    rank, direction, target = ordinal
    result.update(rank=rank, direction=direction, target=target,
                  multi_answer='', multi_audit={}, raw_boxes=[], candidate_boxes=[],
                  refinement_steps=[])
    multi_prompt = f'Locate all the instances that match the following description: {target}.'
    seed = args.seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
    try:
        multi_answer = generate(image, multi_prompt, seed,
                                args.multi_max_new_tokens, args.multi_image_token_limit)
        raw, audit = parse_boxes(multi_answer, image.size)
        result.update(multi_answer=multi_answer, multi_audit=audit, raw_boxes=raw)
        root_unique = deduplicate(raw)
        active = deduplicate(remove_parents(raw))
        current_local, bounds = root_unique, (0, 0, image.width, image.height)
        for depth in range(1, args.max_depth + 1):
            parent_index, reasons = suspicious_parent(current_local,
                result['bbox'] if depth == 1 and valid_box(result['bbox']) else None)
            if parent_index is None:
                break
            parent = map_from_crop(current_local[parent_index], bounds, image.size)
            if parent is None or (depth > 1 and area(parent) < 0.10):
                break
            crop, next_bounds = crop_region(image, parent)
            step = dict(depth=depth, parent=parent, reasons=reasons,
                        crop_bounds=list(next_bounds), accepted_children=[])
            try:
                answer = generate(crop, multi_prompt, seed + depth * 100,
                                  args.multi_max_new_tokens, args.multi_image_token_limit)
                children_local, child_audit = parse_boxes(answer, crop.size)
                step.update(answer=answer, audit=child_audit)
                children = []
                for local in deduplicate(children_local):
                    global_box = map_from_crop(local, next_bounds, image.size)
                    if (global_box and area(global_box) <= 0.8 * area(parent)
                            and intersection(parent, global_box) / area(global_box) >= 0.9):
                        children.append((local, global_box))
                step['accepted_children'] = [global_box for _, global_box in children]
            except Exception as exc:
                step['error'] = f'{type(exc).__name__}: {exc}'
                result['refinement_steps'].append(step)
                break
            result['refinement_steps'].append(step)
            if not children:
                break
            position = next((i for i, box in enumerate(active) if iou(box, parent) >= 0.999), None)
            if position is None:
                active = deduplicate(active + [b for _, b in children])
            else:
                active = deduplicate(active[:position] + [b for _, b in children] + active[position + 1:])
            current_local = [local for local, global_box in children
                             if any(iou(global_box, survivor) >= 0.999 for survivor in active)]
            bounds = next_bounds
            if not current_local:
                break
        result['candidate_boxes'] = active
        choice = select_g(active, rank, direction)
        if choice is not None:
            result.update(status='ok', bbox=choice, source='g_refined' if any(
                step['accepted_children'] for step in result['refinement_steps']) else 'g_initial')
        else:
            result['g_error'] = 'no_candidates' if not active else 'insufficient_candidates'
            if valid_box(result['bbox']):
                result['source'] = 'a_ordinal_fallback'
    except Exception as exc:
        result['g_error'] = f'{type(exc).__name__}: {exc}'
        if valid_box(result['bbox']):
            result['source'] = 'a_ordinal_fallback'
    return result


def load_queries(path, data_root, check_images):
    queries = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(queries, dict) or not queries:
        raise ValueError('queries JSON must be a nonempty ID-keyed object')
    for key, item in queries.items():
        if not isinstance(item, dict) or not isinstance(item.get('query'), str) or not item['query'].strip():
            raise ValueError(f'{key}: missing query')
        relative = item.get('visible')
        if not isinstance(relative, str) or not relative:
            raise ValueError(f'{key}: missing visible path')
        image = Path(relative)
        resolved = (data_root / image).resolve()
        if (image.is_absolute() or '..' in image.parts
                or not resolved.is_relative_to(data_root.resolve())
                or (check_images and not resolved.is_file())):
            raise ValueError(f'{key}: unsafe or missing RGB image: {relative}')
    return list(queries.items())


def export(rows, records, output, full_image_fallback):
    valid, failures, sources = {}, {}, Counter()
    for key, item in rows:
        record = records.get(key)
        if record and record.get('status') == 'ok' and valid_box(record.get('bbox')):
            valid[key] = {**item, 'bbox': record['bbox']}
            sources[record.get('source', 'unknown')] += 1
        elif record:
            failures[key] = record
    pending = len(rows) - len(valid) - len(failures)
    ready = pending == 0 and (not failures or full_image_fallback)
    atomic_json(output / 'predictions_valid.json', valid)
    atomic_json(output / 'failures.json', failures)
    submission = output / 'submission_complete.json'
    if ready:
        atomic_json(submission, {key: valid.get(key, {**item, 'bbox': [0.0, 0.0, 1.0, 1.0]})
                                 for key, item in rows})
    elif submission.exists():
        submission.unlink()
    summary = dict(total_queries=len(rows), ordinal_queries=sum(bool(parse_ordinal(item['query'])) for _, item in rows),
                   model_boxes=len(valid), failed=len(failures), pending=pending,
                   complete=len(valid) == len(rows), submission_ready=ready,
                   fallback_boxes=len(failures) if ready else 0,
                   sources=dict(sources),
                   reversed_corner_boxes=sum(sum(record.get(field, {}).get('reversed_corners', 0)
                       for field in ('a_audit', 'multi_audit')) for record in records.values()),
                   coordinate_format='normalized xyxy, canonical corner order')
    atomic_json(output / 'summary.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--queries', type=Path)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='First N queries; 0 means all')
    parser.add_argument('--only-ordinals', action='store_true', help='Pilot on explicit horizontal ordinals only')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image-token-limit', type=int, default=25600)
    parser.add_argument('--multi-image-token-limit', type=int, default=4096)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--multi-max-new-tokens', type=int, default=2048)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--retries', type=int, default=1)
    parser.add_argument('--max-depth', type=int, default=3)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--fallback-full-image', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--skip-image-check', action='store_true',
                        help='With --check-only, inspect queries when RGB files are unavailable locally')
    args = parser.parse_args(argv)
    if (args.limit < 0 or args.retries < 0 or args.max_depth < 0 or args.seed < 0
            or min(args.image_token_limit, args.multi_image_token_limit,
                   args.max_new_tokens, args.multi_max_new_tokens, args.save_every) < 1
            or args.temperature <= 0 or not 0 < args.top_p <= 1):
        parser.error('invalid numeric settings')
    if args.skip_image_check and not args.check_only:
        parser.error('--skip-image-check is only valid with --check-only')
    args.data_root = args.data_root.resolve()
    args.model_path = args.model_path.resolve()
    queries_path = find_queries(args.data_root, args.queries)
    rows = load_queries(queries_path, args.data_root, not args.skip_image_check)
    total_available = len(rows)
    if args.only_ordinals:
        rows = [(key, item) for key, item in rows if parse_ordinal(item['query'])]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        parser.error('no selected queries')
    ordinal_count = sum(bool(parse_ordinal(item['query'])) for _, item in rows)
    print(f'Validated {len(rows)}/{total_available} queries; {ordinal_count} routed to G', flush=True)
    if args.check_only:
        return 0
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'run.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        config = dict(script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      queries_sha256=hashlib.sha256(queries_path.read_bytes()).hexdigest(),
                      selected_ids=[key for key, _ in rows], data_root=str(args.data_root),
                      model_path=str(args.model_path), settings={name: getattr(args, name) for name in (
                          'seed', 'image_token_limit', 'multi_image_token_limit', 'max_new_tokens',
                          'multi_max_new_tokens', 'temperature', 'top_p', 'retries', 'max_depth')})
        manifest = output / 'run_config.json'
        if manifest.exists():
            if json.loads(manifest.read_text(encoding='utf-8')) != config:
                raise ValueError('run configuration changed; choose a new --output-dir')
        else:
            if (output / 'predictions.jsonl').exists():
                raise ValueError('existing progress without run_config.json')
            atomic_json(manifest, config)
        records = read_progress(output / 'predictions.jsonl')
        summary = export(rows, records, output, args.fallback_full_image)
        if summary['complete']:
            return 0
        generate = make_generator(args)
        start = time.monotonic()
        processed = 0
        with (output / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
            for index, (key, item) in enumerate(rows):
                prior = records.get(key, {})
                if prior.get('status') == 'ok' and valid_box(prior.get('bbox')):
                    continue
                try:
                    record = infer(index, key, item, args, generate)
                except Exception as exc:
                    record = dict(status='error', bbox=None, source='none',
                                  error=f'{type(exc).__name__}: {exc}')
                record['id'] = key
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                records[key] = record
                processed += 1
                if processed % args.save_every == 0 or record['status'] != 'ok':
                    print(f'{index + 1}/{len(rows)} {key} {record["status"]} {record.get("source")} '
                          f'elapsed={time.monotonic() - start:.0f}s', flush=True)
                if processed % args.save_every == 0:
                    export(rows, records, output, args.fallback_full_image)
        summary = export(rows, records, output, args.fallback_full_image)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
        return 0 if summary['submission_ready'] else 2
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
