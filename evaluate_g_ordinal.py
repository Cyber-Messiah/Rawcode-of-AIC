#!/usr/bin/env python3
"""Evaluate deterministic ordinal tile selection (G) from a saved F journal."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from evaluate import box_iou
from ordinal_puzzle_policy import g_choice
from predict_rgb_all import atomic_json
from prepare_ordinal_subset import parse_ordinal
from prepare_reference_subset import load_json, valid_box


def candidates(record, stage):
    if stage == 'initial':
        return record.get('original_boxes', record.get('after_dedup', [])) or []
    return record.get('refined_boxes', record.get('original_boxes',
                                               record.get('after_dedup', []))) or []


def evaluate_g(rows, records, baseline, stage):
    details, predictions = [], {}
    for key, reference in rows:
        record = records[key]
        parsed = parse_ordinal(reference['query'])
        if parsed is None:
            raise ValueError(f'{key}: query is not a supported ordinal format')
        rank, direction, _ = parsed
        boxes = candidates(record, stage)
        index, g_box = g_choice(boxes, rank, direction)
        a_box = baseline[key]['bbox']
        if not valid_box(a_box):
            raise ValueError(f'{key}: invalid A bbox')
        chosen = g_box if g_box is not None else a_box
        source = 'g_tile' if g_box is not None else 'a_puzzle_fallback'
        gt = reference['bbox']
        f_box = record.get('bbox') if record.get('status') in (
            'ok', 'fallback_a_none', 'fallback_a_puzzle') else None
        details.append(dict(id=key, source=source, candidate_count=len(boxes),
                            rank=rank, direction=direction,
                            tile_index=index if index is not None else '',
                            g_iou=box_iou(chosen, gt), a_iou=box_iou(a_box, gt),
                            logged_f_iou=box_iou(f_box, gt) if valid_box(f_box) else 0))
        predictions[key] = {**reference, 'bbox': chosen}
    total = len(details)
    summary = dict(total=total, candidate_stage=stage,
                   g_from_puzzle=sum(row['source'] == 'g_tile' for row in details),
                   a_puzzle_fallbacks=sum(row['source'] == 'a_puzzle_fallback' for row in details),
                   g_mean_iou=statistics.mean(row['g_iou'] for row in details),
                   g_acc_at_05=sum(row['g_iou'] >= .5 for row in details) / total,
                   baseline_mean_iou=statistics.mean(row['a_iou'] for row in details),
                   baseline_acc_at_05=sum(row['a_iou'] >= .5 for row in details) / total,
                   logged_f_mean_iou=statistics.mean(row['logged_f_iou'] for row in details),
                   logged_f_acc_at_05=sum(row['logged_f_iou'] >= .5 for row in details) / total,
                   denominator='all selected references; G uses A if no valid ordinal tile')
    return details, predictions, summary


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path,
                        default=root / 'analysis/ordinal_recursive_gate_v2_20260924/predictions.jsonl')
    parser.add_argument('--references', type=Path,
                        default=root / 'datasets/ordinal_subset/annotations.json')
    parser.add_argument('--baseline-predictions', type=Path,
                        default=root / 'outputs/rgb_all/queries_rgb.json')
    parser.add_argument('--candidate-stage', choices=('initial', 'refined'), default='refined')
    parser.add_argument('--ids', nargs='*')
    parser.add_argument('--output-dir', type=Path, default=root / 'outputs/ordinal_g_replay')
    args = parser.parse_args()
    references = load_json(args.references)
    baseline = load_json(args.baseline_predictions)
    records = {}
    with args.journal.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[record['id']] = record
    selected = list(dict.fromkeys(args.ids)) if args.ids is not None else list(references)
    if not selected:
        parser.error('No selected IDs')
    for key in selected:
        if key not in references or key not in records or key not in baseline:
            raise ValueError(f'{key}: missing reference, journal record, or A prediction')
        if not valid_box(references[key].get('bbox')):
            raise ValueError(f'{key}: invalid reference bbox')
    details, predictions, summary = evaluate_g(
        [(key, references[key]) for key in selected], records, baseline,
        args.candidate_stage)
    summary.update(journal=str(args.journal.resolve()),
                   references=str(args.references.resolve()),
                   baseline_predictions=str(args.baseline_predictions.resolve()))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / 'summary.json', summary)
    atomic_json(args.output_dir / 'queries_g.json', predictions)
    with (args.output_dir / 'per_query.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
