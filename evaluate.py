#!/usr/bin/env python3
"""Evaluate normalized xyxy predictions by reference ID, without GPU dependencies."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from prepare_reference_subset import load_json, valid_box, sha256


def box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0.0


def evaluate(predictions, references):
    if not references:
        raise ValueError('Reference set is empty')
    details = []
    for key, ref in references.items():
        if not isinstance(ref, dict) or not valid_box(ref.get('bbox')):
            raise ValueError(f'Invalid reference bbox: {key}')
        prediction = predictions.get(key)
        status, box = 'missing', None
        if key in predictions:
            box = prediction.get('bbox') if isinstance(prediction, dict) else None
            if isinstance(prediction, dict):
                for field in ('query', 'visible'):
                    if field in prediction and field in ref and prediction[field] != ref[field]:
                        raise ValueError(f'{key}: prediction/reference {field} mismatch')
            if not isinstance(prediction, dict) or not valid_box(box):
                status = 'invalid'
            elif prediction.get('used_fallback') or prediction.get('status', 'ok') != 'ok':
                status = 'failed'
            else:
                status = 'ok'
        iou = box_iou(box, ref['bbox']) if status == 'ok' else 0.0
        details.append(dict(id=key, status=status, iou=iou, prediction=box,
                            reference=ref['bbox'], query=ref.get('query', ''), visible=ref.get('visible', '')))
    scores = [row['iou'] for row in details]
    valid_scores = [row['iou'] for row in details if row['status'] == 'ok']
    image_scores = {}
    for row in details:
        image_scores.setdefault(row['visible'] or row['id'], []).append(row['iou'])
    n = len(details)
    counts = {status: sum(row['status'] == status for row in details)
              for status in ('ok', 'missing', 'invalid', 'failed')}
    summary = dict(reference_count=n, prediction_count=len(predictions),
                   matched_ids=sum(k in predictions for k in references),
                   ignored_unlabeled_predictions=len(set(predictions) - set(references)),
                   counts=counts, valid_prediction_rate=counts['ok']/n,
                   mean_iou=statistics.mean(scores), median_iou=statistics.median(scores),
                   std_iou=statistics.pstdev(scores), min_iou=min(scores), max_iou=max(scores),
                   mean_iou_valid_only=statistics.mean(valid_scores) if valid_scores else None,
                   zero_iou_count=sum(score == 0 for score in scores),
                   accuracy={f'IoU>={t:.2f}': sum(score >= t for score in scores)/n
                             for t in (.25, .5, .75, .9)},
                   image_count=len(image_scores),
                   mean_iou_per_image=statistics.mean(statistics.mean(v) for v in image_scores.values()),
                   denominator='All reference queries; missing/invalid/explicitly failed predictions count as IoU 0',
                   bbox_format='normalized xyxy; threshold accuracy is not detection mAP')
    return summary, details


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=root / 'outputs/rgb_reference/queries_rgb.json')
    parser.add_argument('--references', type=Path, default=root / 'datasets/reference_subset/annotations.json')
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    summary, details = evaluate(load_json(args.predictions), load_json(args.references))
    summary['inputs'] = {name: dict(path=str(path.resolve()), sha256=sha256(path))
                         for name, path in [('predictions', args.predictions), ('references', args.references)]}
    output = args.output_dir or args.predictions.parent / 'evaluation'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'per_query.json').write_text(json.dumps(details, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (output / 'per_query.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(sorted(details, key=lambda row: row['iou']))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
