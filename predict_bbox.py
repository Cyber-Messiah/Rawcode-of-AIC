#!/usr/bin/env python3
"""Generate RGB LocateAnything boxes for a complete competition query set."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import time


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def valid_box(box):
    return (isinstance(box, list) and len(box) == 4
            and all(type(value) in (int, float) and math.isfinite(value)
                    and 0 <= value <= 1 for value in box)
            and box[0] < box[2] and box[1] < box[3])


def parse_box(answer):
    for match in re.finditer(r'<box>(.*?)</box>', answer, re.S):
        body = match.group(1).strip()
        found = (re.fullmatch(r'<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>', body)
                 or re.fullmatch(r'\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?', body))
        if found:
            box = [int(value) / 1000 for value in found.groups()]
            if valid_box(box):
                return box
    return None


def read_progress(path):
    """Repair only a torn final JSONL record; reject corruption elsewhere."""
    records = {}
    if not path.exists():
        return records
    with path.open('rb+') as stream:
        while True:
            start = stream.tell()
            line = stream.readline()
            if not line:
                break
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or 'id' not in row:
                    raise ValueError('Record has no ID')
            except (ValueError, UnicodeDecodeError):
                if not line.endswith(b'\n') and not stream.read(1):
                    stream.seek(start)
                    stream.truncate()
                    break
                raise ValueError(f'Corrupt progress record at byte {start}: {path}')
            records[str(row['id'])] = row
            if not line.endswith(b'\n'):
                stream.write(b'\n')
                break
    return records


def validate_inputs(queries_path, data_root):
    queries = json.loads(queries_path.read_text(encoding='utf-8-sig'))
    if not isinstance(queries, dict) or not queries:
        raise ValueError('queries JSON must be a nonempty object keyed by sample ID')
    for key, item in queries.items():
        if not isinstance(item, dict) or not isinstance(item.get('query'), str) or not item['query'].strip():
            raise ValueError(f'{key}: missing query')
        relative = item.get('visible')
        if not isinstance(relative, str):
            raise ValueError(f'{key}: missing visible image path')
        image = (data_root / relative).resolve()
        if Path(relative).is_absolute() or not image.is_relative_to(data_root) or not image.is_file():
            raise ValueError(f'{key}: missing or unsafe visible image: {relative}')
    return list(queries.items())


def export_results(rows, records, output, allow_fallback):
    successful, failures = {}, {}
    for key, item in rows:
        result = records.get(key)
        if result and result.get('status') == 'ok' and valid_box(result.get('bbox')):
            successful[key] = {**item, 'bbox': result['bbox']}
        elif result:
            failures[key] = result
    atomic_json(output / 'predictions_valid.json', successful)
    atomic_json(output / 'failures.json', failures)
    pending = len(rows) - len(successful) - len(failures)
    complete = len(successful) == len(rows)
    submission_ready = complete or (allow_fallback and pending == 0)
    submission_path = output / 'submission_complete.json'
    if submission_ready:
        submission = {key: successful.get(key, {**item, 'bbox': [0.0, 0.0, 1.0, 1.0]})
                      for key, item in rows}
        atomic_json(submission_path, submission)
    elif submission_path.exists():
        submission_path.unlink()
    summary = dict(total_queries=len(rows), model_boxes=len(successful), failed=len(failures),
                   pending=pending, complete=complete, submission_ready=submission_ready,
                   fallback_boxes=len(failures) if submission_ready and allow_fallback else 0,
                   coordinate_format='normalized xyxy, [x1,y1,x2,y2]')
    atomic_json(output / 'summary.json', summary)
    return summary


def run_rows(rows, records, output, predict, allow_fallback, save_every=50):
    started = time.monotonic()
    processed = 0
    try:
        with (output / 'predictions.jsonl').open('a', encoding='utf-8') as stream:
            for index, (key, item) in enumerate(rows):
                previous = records.get(key, {})
                if previous.get('status') == 'ok' and valid_box(previous.get('bbox')):
                    continue
                try:
                    result = predict(index, key, item)
                except Exception as exc:
                    result = dict(status='error', bbox=None, errors=[f'{type(exc).__name__}: {exc}'])
                result['id'] = key
                stream.write(json.dumps(result, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                records[key] = result
                processed += 1
                if processed % save_every == 0 or result.get('status') != 'ok':
                    print(f'{index + 1}/{len(rows)} {key} {result.get("status")} '
                          f'elapsed={time.monotonic() - started:.0f}s', flush=True)
                if processed % save_every == 0:
                    export_results(rows, records, output, allow_fallback)
    finally:
        summary = export_results(rows, records, output, allow_fallback)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def make_predictor(args):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment')
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

    def predict(index, key, item):
        answer, errors = '', []
        processor.image_processor.in_token_limit = args.image_token_limit
        for attempt in range(args.retries + 1):
            inputs = generated = None
            try:
                seed = args.seed + index * 1009 + attempt
                random.seed(seed)
                np.random.seed(seed % (2 ** 32))
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                with Image.open(args.data_root / item['visible']) as source:
                    image = source.convert('RGB')
                messages = [{'role': 'user', 'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text', 'text': 'Locate a single instance that matches the following description: '
                     + item['query']},
                ]}]
                prompt = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos = processor.process_vision_info(messages)
                inputs = processor(text=[prompt], images=images, videos=videos, return_tensors='pt')
                with torch.inference_mode():
                    generated = model.generate(
                        pixel_values=inputs['pixel_values'].to(device='cuda', dtype=torch.bfloat16),
                        input_ids=inputs['input_ids'].cuda(),
                        attention_mask=inputs['attention_mask'].cuda(),
                        image_grid_hws=torch.as_tensor(inputs['image_grid_hws'], device='cuda'),
                        tokenizer=tokenizer, use_cache=True, max_new_tokens=args.max_new_tokens,
                        generation_mode='hybrid',
                        temperature=args.temperature if attempt == 0 else args.retry_temperature,
                        top_p=args.top_p, do_sample=True)
                answer = generated[0] if isinstance(generated, tuple) else generated
                if not isinstance(answer, str):
                    answer = tokenizer.batch_decode(answer, skip_special_tokens=False)[0]
                box = parse_box(answer)
                if box is not None:
                    return dict(status='ok', bbox=box, answer=answer, attempts=attempt + 1,
                                image_token_limit=processor.image_processor.in_token_limit)
                errors.append('No valid bbox in model output')
            except torch.cuda.OutOfMemoryError:
                errors.append('CUDA out of memory')
                processor.image_processor.in_token_limit = max(256, processor.image_processor.in_token_limit // 2)
            except Exception as exc:
                errors.append(f'{type(exc).__name__}: {exc}')
                break
            finally:
                inputs = generated = None
                gc.collect()
                torch.cuda.empty_cache()
        return dict(status='error', bbox=None, answer=answer, errors=errors)

    return predict


def find_queries(data_root, specified):
    if specified:
        return specified.resolve()
    candidates = [data_root / 'queries' / 'queries.json', data_root / 'queries.json']
    present = [path for path in candidates if path.is_file()]
    if len(present) != 1:
        raise ValueError('Expected one query file at queries/queries.json or queries.json; use --queries')
    return present[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Competition dataset root containing Images/ and queries/')
    parser.add_argument('--queries', type=Path, help='Override query JSON path')
    parser.add_argument('--model-path', required=True, help='Local LocateAnything-3B directory')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='Pilot run only; 0 means all queries')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image-token-limit', type=int, default=25600)
    parser.add_argument('--max-new-tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--retry-temperature', type=float, default=0.2)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--retries', type=int, default=1)
    parser.add_argument('--save-every', type=int, default=50)
    parser.add_argument('--fallback-full-image', action='store_true',
                        help='After retries, use [0,0,1,1] for failures in complete submission')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if (args.limit < 0 or args.retries < 0 or min(args.image_token_limit, args.max_new_tokens, args.save_every) < 1
            or min(args.temperature, args.retry_temperature) < 0 or not 0 < args.top_p <= 1):
        parser.error('Invalid numeric arguments')
    args.data_root = args.data_root.resolve()
    queries_path = find_queries(args.data_root, args.queries)
    rows = validate_inputs(queries_path, args.data_root)
    total_available = len(rows)
    if args.limit:
        rows = rows[:args.limit]
    print(f'Validated {len(rows)} of {total_available} RGB queries; model={args.model_path}', flush=True)
    if args.check_only:
        return 0
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / 'run.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        parser.error(f'{lock} exists; confirm no process is running before removing stale lock')
    os.close(fd)
    try:
        config = dict(query_sha256=hashlib.sha256(queries_path.read_bytes()).hexdigest(),
                      data_root=str(args.data_root), model_path=str(Path(args.model_path).resolve()),
                      limit=args.limit, seed=args.seed, image_token_limit=args.image_token_limit,
                      max_new_tokens=args.max_new_tokens, temperature=args.temperature,
                      retry_temperature=args.retry_temperature, top_p=args.top_p,
                      retries=args.retries)
        manifest = output / 'run_config.json'
        if manifest.exists():
            if json.loads(manifest.read_text(encoding='utf-8')) != config:
                raise ValueError('Configuration changed; use a new --output-dir')
        else:
            if (output / 'predictions.jsonl').exists():
                raise ValueError('Existing progress without run_config.json; use a new --output-dir')
            atomic_json(manifest, config)
        records = read_progress(output / 'predictions.jsonl')
        summary = export_results(rows, records, output, args.fallback_full_image)
        if not summary['complete']:
            summary = run_rows(rows, records, output, make_predictor(args),
                               args.fallback_full_image, args.save_every)
        return 0 if summary['submission_ready'] else 2
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
