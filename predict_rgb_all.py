#!/usr/bin/env python3
"""Resumable RGB grounding. Failures remain explicit and are retried on restart."""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def valid_box(box):
    return (isinstance(box, list) and len(box) == 4
            and all(isinstance(x, (int, float)) and math.isfinite(x) and 0 <= x <= 1 for x in box)
            and box[0] < box[2] and box[1] < box[3])


def parse_box(text):
    for match in re.finditer(r'<box>(.*?)</box>', text, re.S):
        body = match.group(1).strip()
        tagged = re.fullmatch(r'<(\d+)>\s*<(\d+)>\s*<(\d+)>\s*<(\d+)>', body)
        plain = re.fullmatch(r'\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?', body)
        found = tagged or plain
        if found:
            box = [int(x) / 1000 for x in found.groups()]
            if valid_box(box):
                return box
    return None


def read_progress(path):
    """Repair only a torn last record; never silently ignore middle corruption."""
    records = {}
    if not path.exists():
        return records
    with path.open('rb+') as f:
        while True:
            start = f.tell()
            line = f.readline()
            if not line:
                break
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or 'id' not in row:
                    raise ValueError('Progress record has no id')
            except (ValueError, UnicodeDecodeError):
                if not line.endswith(b'\n') and not f.read(1):
                    f.seek(start)
                    f.truncate()
                    break
                raise ValueError(f'Corrupt progress at byte {start}: {path}')
            records[str(row['id'])] = row
            if not line.endswith(b'\n'):
                f.write(b'\n')
                break
    return records


def export_results(rows, records, output):
    good, failed, pending = {}, {}, []
    for key, item in rows:
        row = records.get(key)
        if row and row.get('status') == 'ok' and valid_box(row.get('bbox')):
            good[key] = {**item, 'bbox': row['bbox']}
        elif row:
            failed[key] = row
        else:
            pending.append(key)
    atomic_json(output / 'queries_rgb.json', good)
    atomic_json(output / 'failures.json', failed)
    summary = dict(samples=len(rows), successful=len(good), failed=len(failed),
                   pending=len(pending), complete=len(good) == len(rows),
                   coordinate_format='normalized xyxy, range 0..1')
    atomic_json(output / 'summary.json', summary)
    return summary


def run_rows(rows, records, output, predict, save_every=50):
    started = time.monotonic()
    processed = 0
    try:
        with (output / 'predictions.jsonl').open('a', encoding='utf-8') as f:
            for key, item in rows:
                previous = records.get(key, {})
                if previous.get('status') == 'ok' and valid_box(previous.get('bbox')):
                    continue
                try:
                    row = predict(key, item)
                except Exception as exc:
                    row = dict(status='error', bbox=None, error=f'{type(exc).__name__}: {exc}')
                row['id'] = key
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                f.flush()
                os.fsync(f.fileno())
                records[key] = row
                processed += 1
                print(f'{key}: {row["status"]}; processed this run={processed}; '
                      f'elapsed={time.monotonic() - started:.0f}s', flush=True)
                if processed % save_every == 0:
                    export_results(rows, records, output)
    finally:
        summary = export_results(rows, records, output)
        print(json.dumps(summary, ensure_ascii=False), flush=True)
    return summary


def make_predictor(args):
    import numpy as np
    import torch
    from PIL import Image
    from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
    from predict_locany_competition import build_messages, answer_to_text

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this Python environment.')
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
    processor.image_processor.in_token_limit = args.image_token_limit

    def predict(key, item):
        answer, errors = '', []
        processor.image_processor.in_token_limit = args.image_token_limit
        for attempt in range(args.retries + 1):
            inputs = generated = None
            try:
                # Retry parse failures with AR decoding; lower image budget after OOM.
                seed = args.seed + int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) + attempt
                torch.manual_seed(seed)
                np.random.seed(seed % (2 ** 32))
                with Image.open(args.data_root / item['visible']) as source:
                    rgb = source.convert('RGB')
                messages = build_messages(rgb, str(item['query']))
                text = processor.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                images, videos = processor.process_vision_info(messages)
                inputs = processor(text=[text], images=images, videos=videos, return_tensors='pt')
                with torch.inference_mode():
                    generated = model.generate(
                        pixel_values=inputs['pixel_values'].to(device='cuda', dtype=torch.bfloat16),
                        input_ids=inputs['input_ids'].cuda(), attention_mask=inputs['attention_mask'].cuda(),
                        image_grid_hws=torch.as_tensor(inputs['image_grid_hws'], device='cuda'),
                        tokenizer=tokenizer, use_cache=True, max_new_tokens=args.max_new_tokens,
                        generation_mode='hybrid' if attempt == 0 else 'slow',
                        do_sample=False, temperature=1.0)
                answer = answer_to_text(generated, tokenizer)
                box = parse_box(answer)
                if box:
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


def main():
    root = Path(__file__).resolve().parents[2]
    datasets = list((root / 'LocateAnything_Competition').glob('*/queries/queries.json'))
    default_annotation = datasets[0] if len(datasets) == 1 else None
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotation', type=Path, default=default_annotation)
    p.add_argument('--data-root', type=Path)
    p.add_argument('--model-path', default=str(root / 'LocateAnything-3B'))
    p.add_argument('--output-dir', type=Path, default=root / 'outputs' / 'rgb_all')
    p.add_argument('--limit', type=int, default=0, help='0 = all queries')
    p.add_argument('--image-token-limit', type=int, default=4096)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--retries', type=int, default=2)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every', type=int, default=50)
    p.add_argument('--check-only', action='store_true', help='Validate input paths without loading the model')
    args = p.parse_args()
    if args.annotation is None:
        p.error('Specify --annotation')
    if args.limit < 0 or args.retries < 0 or min(args.save_every, args.max_new_tokens, args.image_token_limit) < 1:
        p.error('Invalid numeric arguments')
    args.data_root = (args.data_root or args.annotation.parent.parent).resolve()
    source = args.annotation.read_bytes()
    data = json.loads(source)
    if not isinstance(data, dict) or not data:
        p.error('Expected a nonempty query object keyed by sample id')
    rows = list(data.items())[:args.limit or None]
    for key, item in rows:
        if not isinstance(item.get('query'), str) or not item['query'].strip():
            p.error(f'{key}: missing query')
        if not item.get('visible') or not (args.data_root / item['visible']).is_file():
            p.error(f'{key}: missing RGB image')
    print(f'Validated {len(rows)} RGB queries.', flush=True)
    if args.check_only:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive ownership prevents two processes from corrupting the same journal.
    lock = args.output_dir / 'run.lock'
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        p.error(f'{lock} exists. If the previous process has stopped, remove this stale lock.')
    os.close(fd)
    try:
        manifest = dict(annotation_sha256=hashlib.sha256(source).hexdigest(),
                        data_root=str(args.data_root), model_path=str(Path(args.model_path).resolve()),
                        image_token_limit=args.image_token_limit, max_new_tokens=args.max_new_tokens,
                        seed=args.seed, pipeline_version=1)
        manifest_path = args.output_dir / 'run_config.json'
        if manifest_path.exists():
            if json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
                raise ValueError('Run configuration changed: use a different --output-dir.')
        else:
            if (args.output_dir / 'predictions.jsonl').exists():
                raise ValueError('Unidentified existing progress: use a new --output-dir.')
            atomic_json(manifest_path, manifest)
        records = read_progress(args.output_dir / 'predictions.jsonl')
        summary = export_results(rows, records, args.output_dir)
        if not summary['complete']:
            predict = make_predictor(args)
            summary = run_rows(rows, records, args.output_dir, predict, args.save_every)
        return 0 if summary['complete'] else 2
    finally:
        lock.unlink()


if __name__ == '__main__':
    raise SystemExit(main())
