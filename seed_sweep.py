#!/usr/bin/env python3
"""Repeated RGB evaluation, retaining aggregate statistics only (no predictions)."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import statistics
import time

from evaluate import evaluate
from predict_rgb_all import atomic_json, make_predictor
from prepare_reference_subset import load_json, sha256, valid_box


def aggregate(runs):
    values = {}
    for metric in ('mean_iou', 'median_iou', 'valid_prediction_rate', 'acc_025', 'acc_05', 'acc_075', 'acc_09'):
        series = [r[metric] for r in runs]
        values[metric] = dict(mean=statistics.mean(series),
                              sample_std=statistics.stdev(series) if len(series) > 1 else None,
                              min=min(series), max=max(series))
    return values


def run_once(rows, references, predict, seed, progress_every=50):
    predictions, errors = {}, Counter()
    started = time.monotonic()
    for i, (key, item) in enumerate(rows, 1):
        try:
            result = predict(key, item)
        except Exception as exc:
            result = dict(status='error', bbox=None, errors=[type(exc).__name__])
        predictions[key] = {k: result[k] for k in ('bbox', 'status') if k in result}
        if result.get('status') != 'ok':
            for error in result.get('errors', ['prediction failed']):
                errors[str(error).split(':', 1)[0]] += 1
        if i % progress_every == 0 or i == len(rows):
            print(f'seed={seed} progress={i}/{len(rows)} elapsed={time.monotonic()-started:.0f}s', flush=True)
    summary, _ = evaluate(predictions, references)
    return dict(seed=seed, samples=len(rows), mean_iou=summary['mean_iou'],
                median_iou=summary['median_iou'], valid_prediction_rate=summary['valid_prediction_rate'],
                acc_025=summary['accuracy']['IoU>=0.25'], acc_05=summary['accuracy']['IoU>=0.50'],
                acc_075=summary['accuracy']['IoU>=0.75'], acc_09=summary['accuracy']['IoU>=0.90'],
                zero_iou_count=summary['zero_iou_count'], counts=summary['counts'],
                error_counts=dict(errors), elapsed_seconds=time.monotonic()-started)


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--annotation', type=Path, default=root/'datasets/reference_subset/queries.json')
    p.add_argument('--references', type=Path, default=root/'datasets/reference_subset/annotations.json')
    p.add_argument('--data-root', type=Path)
    p.add_argument('--model-path', default=str(root/'LocateAnything-3B'))
    p.add_argument('--output-dir', type=Path, default=root/'outputs/seed_sweep')
    p.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44, 45, 46])
    p.add_argument('--temperature', type=float, default=1.0,
                   help='LocateAnything uses temperature>0 for sampling; 0 is greedy')
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--generation-mode', choices=['hybrid', 'slow', 'fast'], default='hybrid')
    p.add_argument('--image-token-limit', type=int, default=4096)
    p.add_argument('--max-new-tokens', type=int, default=128)
    p.add_argument('--retries', type=int, default=2)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--check-only', action='store_true')
    p.add_argument('--original-rgb', action='store_true',
                   help='Use original predict_locany_competition.py RGB settings')
    p.add_argument('--full-queries', type=Path, default=root/'datasets/full/queries.json',
                   help='Original 9555-query order, used for historical per-sample seeds')
    args = p.parse_args()
    if args.original_rgb:
        args.image_token_limit = 25600
        args.temperature = 0.7
        args.top_p = 0.9
        args.retries = 1
        args.generation_mode = 'hybrid'
        args.retry_generation_mode = 'hybrid'
        args.retry_temperature = 0.2
        args.seed_scheme = 'original_index'
        original = load_json(args.full_queries)
        args.sample_indices = {key: index for index, key in enumerate(original)}
    else:
        args.retry_generation_mode = 'slow'
        args.retry_temperature = args.temperature
        args.seed_scheme = 'id_hash'
    if (len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0 or max(args.seeds) >= 2**32
            or not 0 <= args.temperature < float('inf') or not 0 < args.top_p <= 1
            or args.limit < 0 or args.retries < 0 or min(args.image_token_limit, args.max_new_tokens) < 1):
        p.error('Invalid seeds or generation parameters')
    args.data_root = (args.data_root or args.annotation.parent).resolve()
    queries, references = load_json(args.annotation), load_json(args.references)
    rows = [(key, item) for key, item in queries.items() if key in references][:args.limit or None]
    if not rows:
        p.error('No reference questions matched')
    for key, item in rows:
        if args.seed_scheme == 'original_index' and key not in args.sample_indices:
            p.error(f'Query missing from original 9555-query order: {key}')
        if not valid_box(references[key].get('bbox')):
            p.error(f'Invalid reference: {key}')
        for field in ('visible', 'query'):
            if item.get(field) != references[key].get(field):
                p.error(f'Reference mismatch: {key}/{field}')
        if not (args.data_root/item['visible']).is_file():
            p.error(f'Missing image: {key}')
    references = {key: references[key] for key, _ in rows}
    print(f'{len(rows)} reference questions; seeds={args.seeds}; temperature={args.temperature}', flush=True)
    if args.check_only:
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir/'run.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    try:
        config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                  if k not in ('check_only', 'output_dir', 'sample_indices')}
        config['query_sha256'] = sha256(args.annotation)
        config['reference_sha256'] = sha256(args.references)
        if args.original_rgb:
            config['full_queries_sha256'] = sha256(args.full_queries)
        config['predictor_sha256'] = sha256(Path(__file__).with_name('predict_rgb_all.py'))
        config['sweep_sha256'] = sha256(Path(__file__))
        model_dir = Path(args.model_path)
        config['model_files'] = {str(f.relative_to(model_dir)): dict(size=f.stat().st_size, mtime_ns=f.stat().st_mtime_ns)
                                 for f in model_dir.rglob('*') if f.is_file() and '.cache' not in f.parts}
        report_path = args.output_dir/'statistics.json'
        report = dict(config=config, runs=[], complete=False)
        if report_path.exists():
            report = load_json(report_path)
            if report['config'] != config:
                raise ValueError('Configuration/model changed; use a new --output-dir')
        atomic_json(report_path, report)
        done = {r['seed'] for r in report['runs']}
        pending = [seed for seed in args.seeds if seed not in done]
        if pending:
            args.seed = pending[0]
            predict = make_predictor(args)  # Load model once; closure reads current args.seed.
            import torch, transformers
            report['environment'] = dict(torch=torch.__version__, transformers=transformers.__version__,
                                          cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0))
            for seed in pending:
                args.seed = seed
                run = run_once(rows, references, predict, seed)
                report['runs'].append(run)
                report['aggregate'] = aggregate(report['runs'])
                report['complete'] = len(report['runs']) == len(args.seeds)
                atomic_json(report_path, report)
                print(json.dumps(run, ensure_ascii=False), flush=True)
        print(json.dumps(report.get('aggregate', {}), indent=2), flush=True)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
