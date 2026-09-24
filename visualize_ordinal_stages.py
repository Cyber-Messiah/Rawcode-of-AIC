#!/usr/bin/env python3
"""Render saved A/F ordinal predictions for manual, GPU-free error review."""
import argparse
import csv
from collections import Counter
from html import escape
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from evaluate import box_iou
from experiment_f_official_multi import deduplicate, make_puzzle, parse_boxes, remove_parent_boxes
from prepare_reference_subset import load_json, valid_box


COLORS = dict(gt='#36e04c', a='#ff9933', final='#fa4ee2', raw='#3ec9ff',
              best='#ffe049', candidate='#40b7ed', selected='#fa4756',
              expected='#ffbf3c', parent='#fa4756', child='#ff9f43')
CATEGORIES = ('accurate_candidate', 'coarse_contains_gt', 'partial_overlap')


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def overlap(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def classify(raw, gt):
    """Separate selection errors from oversized and partially aligned raw boxes."""
    if not raw:
        return 'no_raw_boxes'
    if any(box_iou(box, gt) >= .5 for box in raw):
        return 'accurate_candidate'
    if any(overlap(box, gt) / area(gt) >= .9 for box in raw):
        return 'coarse_contains_gt'
    if any(overlap(box, gt) / area(gt) >= .5 or box_iou(box, gt) >= .1 for box in raw):
        return 'partial_overlap'
    return 'no_meaningful_overlap'


def font_for(image):
    size = max(13, round(min(image.size) / 75))
    for name in ('C:/Windows/Fonts/arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def rectangle(draw, image, box, color, label, font, inset=0):
    if (not isinstance(box, (list, tuple)) or len(box) != 4
            or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
            or box[0] >= box[2] or box[1] >= box[3]
            or box[2] <= 0 or box[0] >= 1 or box[3] <= 0 or box[1] >= 1):
        return
    width, height = image.size
    x1 = max(0, min(width - 1, round(box[0] * width) + inset))
    y1 = max(0, min(height - 1, round(box[1] * height) + inset))
    x2 = max(x1, min(width - 1, round(box[2] * width) - inset))
    y2 = max(y1, min(height - 1, round(box[3] * height) - inset))
    stroke = max(2, round(min(width, height) / 280))
    draw.rectangle((x1, y1, x2, y2), outline=color, width=stroke)
    if label:
        bbox = draw.textbbox((0, 0), label, font=font)
        label_width = bbox[2] - bbox[0] + 8
        label_height = bbox[3] - bbox[1] + 6
        tx = min(x1, max(0, width - label_width))
        ty = y1 if y1 + label_height < height else max(0, height - label_height)
        draw.rectangle((tx, ty, tx + label_width, ty + label_height), fill='#111827')
        draw.text((tx + 4, ty + 2 - bbox[1]), label, fill=color, font=font)


def save_overlay(image, path, boxes):
    marked = image.copy()
    draw = ImageDraw.Draw(marked)
    font = font_for(marked)
    for box, color, label in boxes:
        rectangle(draw, marked, box, color, label, font)
    marked.save(path)


def puzzle_image(image, boxes, selection, gt, path, tile_height, gap, max_width):
    ordered = sorted(boxes, key=lambda b: (b[0] + b[2]) / 2)
    if not ordered:
        return None
    puzzle, spans, used = make_puzzle(image, ordered, tile_height, gap, max_width)
    if puzzle is None:
        return None
    marked = puzzle.copy()
    draw = ImageDraw.Draw(marked)
    font = font_for(marked)
    for index, (left, right) in enumerate(spans):
        tile_box = [left, 0, right, 1]
        if box_iou(used[index], gt) >= .5:
            rectangle(draw, marked, tile_box, COLORS['gt'], '', font, inset=2)
        if index == selection.get('selected_index'):
            rectangle(draw, marked, tile_box, COLORS['selected'], '', font, inset=7)
        rectangle(draw, marked, tile_box, COLORS['candidate'], str(index + 1), font)
    if valid_box(selection.get('puzzle_box')):
        rectangle(draw, marked, selection['puzzle_box'], COLORS['final'], 'model', font)
    marked.save(path)
    return list(puzzle.size)


def crop_coordinates(global_box, bounds, size):
    left, top, right, bottom = bounds
    width, height = size
    return [(global_box[0] * width - left) / (right - left),
            (global_box[1] * height - top) / (bottom - top),
            (global_box[2] * width - left) / (right - left),
            (global_box[3] * height - top) / (bottom - top)]


def render_case(key, record, gt, a_box, image, case_dir, args, category):
    case_dir.mkdir(parents=True, exist_ok=True)
    raw = record.get('raw_boxes') or []
    initial = record.get('original_selection') or {}
    final = record.get('bbox') if record.get('status') in ('ok', 'fallback_a_none') else None
    image.save(case_dir / '00_rgb.png')
    save_overlay(image, case_dir / '01_overview.png',
                 [(a_box, COLORS['a'], 'A'), (final, COLORS['final'], 'final F'),
                  (gt, COLORS['gt'], 'GT')])

    best = max(range(len(raw)), key=lambda i: box_iou(raw[i], gt)) if raw else None
    raw_draw = [(box, COLORS['best'] if i == best else COLORS['raw'], f'raw {i + 1}')
                for i, box in enumerate(raw)]
    raw_draw.append((gt, COLORS['gt'], 'GT'))
    save_overlay(image, case_dir / '02_raw_multi.png', raw_draw)

    after_parent = remove_parent_boxes(raw, args.parent_containment,
                                       args.parent_min_children, args.parent_min_area_ratio,
                                       args.dedup_iou)
    original = record.get('original_boxes') or []
    if deduplicate(after_parent, args.dedup_iou) != original:
        raise ValueError(f'{key}: candidate filtering does not reproduce the inference log')
    save_overlay(image, case_dir / '03_after_parent.png',
                 [(box, COLORS['candidate'], f'kept {i + 1}') for i, box in
                  enumerate(after_parent)] + [(gt, COLORS['gt'], 'GT')])
    ordered = sorted(original, key=lambda b: (b[0] + b[2]) / 2)
    candidate_draw = [(box, COLORS['candidate'], f'L{i + 1}')
                      for i, box in enumerate(ordered)]
    candidate_draw.extend([(initial.get('bbox'), COLORS['selected'], 'F initial'),
                           (gt, COLORS['gt'], 'GT')])
    save_overlay(image, case_dir / '04_after_dedup.png', candidate_draw)
    initial_size = puzzle_image(image, original, initial, gt,
                                case_dir / '05_puzzle_initial.png', args.tile_height,
                                args.gap, args.max_puzzle_width)
    if initial_size and initial.get('puzzle_size') and initial_size != initial['puzzle_size']:
        raise ValueError(f'{key}: reconstructed initial puzzle size differs from inference log')

    stage_images = ['00_rgb.png', '01_overview.png', '02_raw_multi.png',
                    '03_after_parent.png', '04_after_dedup.png']
    if initial_size:
        stage_images.append('05_puzzle_initial.png')
    for step in record.get('refinement_steps') or []:
        depth = step['depth']
        bounds = step['crop_bounds']
        crop = image.crop(tuple(bounds))
        raw_children = parse_boxes(step.get('answer', ''))
        crop_draw = [(box, COLORS['raw'], f'raw child {i + 1}')
                     for i, box in enumerate(raw_children)]
        crop_draw.extend((crop_coordinates(box, bounds, image.size), COLORS['child'],
                          f'accepted {i + 1}')
                         for i, box in enumerate(step.get('accepted_children') or []))
        crop_draw.append((crop_coordinates(gt, bounds, image.size), COLORS['gt'], 'GT'))
        crop_name = f'06_refine_{depth:02d}_crop.png'
        save_overlay(crop, case_dir / crop_name, crop_draw)
        full_name = f'06_refine_{depth:02d}_full.png'
        save_overlay(image, case_dir / full_name,
                     [(step['parent'], COLORS['parent'], 'parent'),
                      *[(box, COLORS['child'], f'child {i + 1}') for i, box in
                        enumerate(step.get('accepted_children') or [])],
                      (gt, COLORS['gt'], 'GT')])
        stage_images.extend([crop_name, full_name])

    refined = record.get('refined_selection') or {}
    if refined:
        refined_boxes = record.get('refined_boxes') or []
        save_overlay(image, case_dir / '07_refined_candidates.png',
                     [(box, COLORS['candidate'], f'L{i + 1}') for i, box in
                      enumerate(sorted(refined_boxes, key=lambda b: (b[0] + b[2]) / 2))]
                     + [(refined.get('bbox'), COLORS['selected'], 'F refined'),
                        (gt, COLORS['gt'], 'GT')])
        stage_images.append('07_refined_candidates.png')
        refined_size = puzzle_image(image, refined_boxes, refined, gt,
                                    case_dir / '08_puzzle_refined.png', args.tile_height,
                                    args.gap, args.max_puzzle_width)
        if refined_size and refined.get('puzzle_size') and refined_size != refined['puzzle_size']:
            raise ValueError(f'{key}: reconstructed refined puzzle size differs from inference log')
        if refined_size:
            stage_images.append('08_puzzle_refined.png')

    captions = {
        '00_rgb.png': '未经标记的完整 RGB 原图',
        '01_overview.png': '原图：人工答案 GT（绿）、A（橙）、最终 F（紫）',
        '02_raw_multi.png': 'F 第一阶段原始多框（蓝）；与 GT IoU 最高的原始框为黄',
        '03_after_parent.png': '去父框后的候选，保留原始输出顺序',
        '04_after_dedup.png': '进一步去重后的拼图候选（L 为从左往右编号）；初始 F 选择为红',
        '05_puzzle_initial.png': '重建初始拼图：绿框为 IoU≥0.5 的候选，红框为模型所选，紫框为模型在拼图上的输出',
        '07_refined_candidates.png': '细分后候选与重新选择的框',
        '08_puzzle_refined.png': '重建细分后拼图',
    }
    for step in record.get('refinement_steps') or []:
        depth = step['depth']
        captions[f'06_refine_{depth:02d}_crop.png'] = f'第 {depth} 层裁剪图：原始子框（蓝）、接受的子框（橙）、GT（绿）'
        captions[f'06_refine_{depth:02d}_full.png'] = f'第 {depth} 层映射回原图：父框（红）、接受的子框（橙）、GT（绿）'
    figures = '\n'.join(f'<figure><figcaption>{escape(captions[name])}</figcaption>'
                        f'<img loading="lazy" src="{name}"></figure>' for name in stage_images)
    answers = [('第一阶段提示', record.get('multi_prompt', '')),
               ('第一阶段原始回答', record.get('multi_answer', '')),
               ('初始拼图提示', initial.get('prompt', '')),
               ('初始拼图回答', initial.get('answer', '')),
               ('细分拼图提示', refined.get('prompt', '')),
               ('细分拼图回答', refined.get('answer', ''))]
    for step in record.get('refinement_steps') or []:
        answers.append((f"第 {step['depth']} 层裁剪回答", step.get('answer', step.get('error', ''))))
    answer_html = '\n'.join(f'<details><summary>{escape(label)}</summary><pre>{escape(value)}</pre></details>'
                            for label, value in answers if value)
    title = f'{key} · {record["query"]}'
    metrics = (f'类别：{category} | 候选数：原始 {len(raw)} → 初始 {len(original)} '
               f'→ 细分后 {len(record.get("refined_boxes") or [])} | '
               f'A IoU={box_iou(a_box, gt):.3f} | '
               f'初始 F IoU={box_iou(initial["bbox"], gt) if valid_box(initial.get("bbox")) else 0:.3f} | '
               f'最终 F IoU={box_iou(final, gt) if valid_box(final) else 0:.3f}')
    html = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>{escape(title)}</title>
<style>body{{font:16px/1.5 sans-serif;max-width:1500px;margin:24px auto;padding:0 16px;background:#111827;color:#f3f4f6}}
a{{color:#93c5fd}}figure{{margin:24px 0;padding:16px;background:#1f2937;border-radius:10px}}
figcaption{{margin-bottom:12px;font-weight:bold}}img{{display:block;max-width:100%;height:auto}}
pre{{white-space:pre-wrap;overflow-wrap:anywhere}}details{{padding:8px;border-bottom:1px solid #374151}}</style>
<p><a href="../index.html">← 全部案例</a></p><h1>{escape(title)}</h1>
<p>{escape(metrics)}</p><p>图片中的 GT 只用于人工核查，不参与预测。编号 L 表示原图中从左至右的候选顺序。</p>
{figures}<h2>模型文本记录</h2>{answer_html}</html>'''
    (case_dir / 'index.html').write_text(html, encoding='utf-8')
    return dict(id=key, category=category, query=record['query'], raw_count=len(raw),
                initial_count=len(original), refined_count=len(record.get('refined_boxes') or []),
                refinement_calls=len(record.get('refinement_steps') or []),
                a_iou=round(box_iou(a_box, gt), 4),
                initial_f_iou=round(box_iou(initial['bbox'], gt), 4) if valid_box(initial.get('bbox')) else 0,
                final_f_iou=round(box_iou(final, gt), 4) if valid_box(final) else 0,
                image_count=len(stage_images))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal', type=Path,
                        default=root / 'analysis/ordinal_recursive_gate_v2_20260924/predictions.jsonl')
    parser.add_argument('--references', type=Path,
                        default=root / 'datasets/ordinal_subset/annotations.json')
    parser.add_argument('--baseline-predictions', type=Path,
                        default=root / 'outputs/rgb_all/queries_rgb.json')
    complete_rgb = root / 'analysis/ordinal_recursive_gate_v2_20260924/rgb_complete'
    parser.add_argument('--data-root', type=Path,
                        default=complete_rgb if complete_rgb.is_dir() else root / 'datasets/reference_subset')
    parser.add_argument('--output-dir', type=Path, default=root / 'outputs/ordinal_stage_review_45')
    parser.add_argument('--ids', nargs='*', help='Explicit IDs; bypass the default 45-case filter')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--list-only', action='store_true')
    parser.add_argument('--tile-height', type=int, default=224)
    parser.add_argument('--gap', type=int, default=12)
    parser.add_argument('--max-puzzle-width', type=int, default=1536)
    parser.add_argument('--parent-containment', type=float, default=.9)
    parser.add_argument('--parent-min-children', type=int, default=2)
    parser.add_argument('--parent-min-area-ratio', type=float, default=1.5)
    parser.add_argument('--dedup-iou', type=float, default=.5)
    args = parser.parse_args()
    if (args.limit < 0 or min(args.tile_height, args.max_puzzle_width,
                              args.parent_min_children) < 1 or args.gap < 0
            or not 0 < args.parent_containment <= 1 or args.parent_min_area_ratio <= 1
            or not 0 <= args.dedup_iou <= 1):
        parser.error('Invalid numeric arguments')
    refs = load_json(args.references)
    baseline = load_json(args.baseline_predictions)
    records = {}
    with args.journal.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                record = json.loads(line)
                records[record['id']] = record
    if args.ids:
        missing = set(args.ids) - set(records)
        if missing:
            parser.error(f'IDs absent from journal: {sorted(missing)}')
        selected = list(dict.fromkeys(args.ids))
    else:
        selected = []
        for key, record in records.items():
            if key not in refs or key not in baseline:
                continue
            gt = refs[key]['bbox']
            a_box = baseline[key]['bbox']
            final = record.get('bbox')
            if (box_iou(a_box, gt) < .5
                    and (not valid_box(final) or box_iou(final, gt) < .5)
                    and classify(record.get('raw_boxes') or [], gt) in CATEGORIES):
                selected.append(key)
    selected = selected[:args.limit or None]
    counts = Counter(classify(records[key].get('raw_boxes') or [], refs[key]['bbox']) for key in selected)
    print(f'Selected {len(selected)} cases: {dict(counts)}', flush=True)
    if args.list_only:
        for key in selected:
            print(key, classify(records[key].get('raw_boxes') or [], refs[key]['bbox']))
        return 0
    if not selected:
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, key in enumerate(selected, 1):
        record = records[key]
        relative = Path(record['visible'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'{key}: unsafe image path')
        image_path = args.data_root / relative
        try:
            with Image.open(image_path) as source:
                image = source.convert('RGB')
        except OSError as exc:
            raise OSError(f'{key}: cannot decode {image_path}; pass --data-root with complete RGB images') from exc
        gt = refs[key]['bbox']
        a_box = baseline[key]['bbox']
        category = classify(record.get('raw_boxes') or [], gt)
        row = render_case(key, record, gt, a_box, image,
                          args.output_dir / key, args, category)
        manifest.append(row)
        print(f'{index}/{len(selected)} {key}: {category}, {row["image_count"]} images', flush=True)
    with (args.output_dir / 'manifest.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    sections = []
    for category in CATEGORIES:
        cases = [row for row in manifest if row['category'] == category]
        links = '\n'.join(f'<li><a href="{row["id"]}/index.html">{row["id"]}</a> '
                          f'({row["raw_count"]} raw) — {escape(row["query"])}</li>' for row in cases)
        sections.append(f'<h2>{category} ({len(cases)})</h2><ol>{links}</ol>')
    index = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>A/F 序数失败案例</title>
<style>body{{font:16px/1.6 sans-serif;max-width:1100px;margin:30px auto;padding:0 18px;background:#111827;color:#f3f4f6}}
a{{color:#93c5fd}}li{{margin:5px 0}}</style><h1>A/F 都错但 F 有部分候选信息的案例</h1>
<p>共 {len(manifest)} 题。三类依次为：准确候选却未选对；大框覆盖答案；部分覆盖或偏移。</p>
{''.join(sections)}</html>'''
    (args.output_dir / 'index.html').write_text(index, encoding='utf-8')
    print(f'Open {args.output_dir / "index.html"}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
