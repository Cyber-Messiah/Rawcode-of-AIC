#!/usr/bin/env python3
"""Render changed A/G/G-prime predictions with a separate, magnified GT panel."""

import argparse
import csv
import json
import math
from collections import Counter
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent

COLORS = {'A': '#ffab47', 'G': '#dc72ff', "G′": '#3dc9ff', 'GT': '#65f17a'}
GROUPS = [('improved', '01 · 变对'), ('worsened', '02 · 变错'),
          ('still_wrong', '03 · 仍错但框已改变'),
          ('still_correct', '04 · 仍对但框已改变')]
PANEL_W, PANEL_H = 466, 335
PANEL_GAP, LEFT = 14, 20
CANVAS_W = LEFT * 2 + PANEL_W * 4 + PANEL_GAP * 3


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return {row['id']: row for row in csv.DictReader(stream)}


def font(size):
    for path in ('C:/Windows/Fonts/msyh.ttc', 'C:/Windows/Fonts/arial.ttf',
                 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


FONTS = {size: font(size) for size in (16, 19, 22, 26)}


def draw_text(draw, xy, value, size=19, fill='#edf3fb'):
    draw.text(xy, str(value), font=FONTS[size], fill=fill)


def box_iou(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    common = max(0, right-left) * max(0, bottom-top)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    return common/(area_a+area_b-common) if area_a+area_b-common > 0 else 0


def load_image(relative, data_root):
    rel = Path(relative)
    if rel.is_absolute() or '..' in rel.parts:
        raise ValueError(f'Unsafe RGB path: {relative}')
    path = data_root / rel
    if not path.is_file():
        raise FileNotFoundError(f'RGB image is missing: {path}')
    with Image.open(path) as image:
        image.load()  # Fail explicitly on truncated files.
        return image.convert('RGB'), path


def gt_crop(gt):
    """A GT-centered crop where even a tiny GT box has visible pixel width."""
    cx, cy = (gt[0]+gt[2])/2, (gt[1]+gt[3])/2
    span_x = min(1, max(.10, (gt[2]-gt[0])*5))
    span_y = min(1, max(.10, (gt[3]-gt[1])*5))
    x1 = max(0, min(1-span_x, cx-span_x/2))
    y1 = max(0, min(1-span_y, cy-span_y/2))
    return (x1, y1, x1+span_x, y1+span_y)


def pixels(box, crop, offset, display_size):
    x1, y1, x2, y2 = crop
    sx = display_size[0]/(x2-x1)
    sy = display_size[1]/(y2-y1)
    return (offset[0] + round((box[0]-x1)*sx),
            offset[1] + round((box[1]-y1)*sy),
            offset[0] + round((box[2]-x1)*sx),
            offset[1] + round((box[3]-y1)*sy))


def visible(rect, bounds):
    return (rect[2] > bounds[0] and rect[0] < bounds[2]
            and rect[3] > bounds[1] and rect[1] < bounds[3])


def outline(draw, rect, color, dashed=False, label=None, bounds=None):
    if bounds and not visible(rect, bounds):
        return False
    if bounds:
        rect = (max(bounds[0], rect[0]), max(bounds[1], rect[1]),
                min(bounds[2], rect[2]), min(bounds[3], rect[3]))
    if rect[2] <= rect[0] or rect[3] <= rect[1]:
        return False
    if dashed:
        # Dark underlay and bright GT dashes stay legible over light images.
        for x in range(rect[0], rect[2], 15):
            segment = (x, rect[1], min(x+9, rect[2]), rect[1])
            draw.line(segment, fill='#101923', width=9)
            draw.line(segment, fill=color, width=5)
            segment = (x, rect[3], min(x+9, rect[2]), rect[3])
            draw.line(segment, fill='#101923', width=9)
            draw.line(segment, fill=color, width=5)
        for y in range(rect[1], rect[3], 15):
            segment = (rect[0], y, rect[0], min(y+9, rect[3]))
            draw.line(segment, fill='#101923', width=9)
            draw.line(segment, fill=color, width=5)
            segment = (rect[2], y, rect[2], min(y+9, rect[3]))
            draw.line(segment, fill='#101923', width=9)
            draw.line(segment, fill=color, width=5)
    else:
        draw.rectangle(rect, outline='#101923', width=10)
        draw.rectangle(rect, outline=color, width=5)
    if label:
        font_used = FONTS[16]
        text_bounds = draw.textbbox((0, 0), label, font=font_used)
        tw, th = text_bounds[2]-text_bounds[0], text_bounds[3]-text_bounds[1]
        lx = max(bounds[0] if bounds else 0, rect[0])
        ly = max(bounds[1] if bounds else 0, rect[1]-th-9)
        draw.rectangle((lx, ly, lx+tw+10, ly+th+7), fill='#101923')
        draw.text((lx+5, ly+2-text_bounds[1]), label, font=font_used, fill=color)
    return True


def panel(canvas, source, x, y, name, box, gt, crop, zoom, iou):
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((x, y, x+PANEL_W, y+PANEL_H), radius=8,
                           fill='#1c2939', outline='#31465d', width=2)
    header = f'{name}  IoU {iou:.3f}' if name != 'GT' else 'GT  人工答案'
    draw_text(draw, (x+12, y+6), header, 22, COLORS[name])
    region = (x+9, y+36, x+PANEL_W-9, y+PANEL_H-12)
    src_w, src_h = source.size
    left = max(0, min(src_w-1, math.floor(crop[0]*src_w)))
    top = max(0, min(src_h-1, math.floor(crop[1]*src_h)))
    right = max(left+1, min(src_w, math.ceil(crop[2]*src_w)))
    bottom = max(top+1, min(src_h, math.ceil(crop[3]*src_h)))
    picture = source.crop((left, top, right, bottom))
    picture.thumbnail((region[2]-region[0], region[3]-region[1]), Image.Resampling.LANCZOS)
    px = region[0] + (region[2]-region[0]-picture.width)//2
    py = region[1] + (region[3]-region[1]-picture.height)//2
    canvas.paste(picture, (px, py))
    # Use the actual integer pixel crop to keep both box transforms exact.
    actual_crop = (left/src_w, top/src_h, right/src_w, bottom/src_h)
    bounds = (px, py, px+picture.width-1, py+picture.height-1)
    draw = ImageDraw.Draw(canvas)
    if name != 'GT':
        shown = outline(draw, pixels(box, actual_crop, (px, py), picture.size),
                        COLORS[name], label=name, bounds=bounds)
        if zoom and not shown:
            draw.rounded_rectangle((px+6, py+6, px+220, py+35), radius=4, fill='#101923')
            draw_text(draw, (px+12, py+7), '预测框在此局部视野外', 16, COLORS[name])
    gt_rect = pixels(gt, actual_crop, (px, py), picture.size)
    outline(draw, gt_rect, COLORS['GT'], dashed=name != 'GT', label='GT', bounds=bounds)


def render_case(case, image, path):
    canvas = Image.new('RGB', (CANVAS_W, 806), '#101923')
    draw = ImageDraw.Draw(canvas)
    heading = (f"{case['id']}  {case['group_label']}  |  "
               f"A {case['a_iou']:.3f}  G {case['g_iou']:.3f}  G′ {case['new_iou']:.3f}")
    draw_text(draw, (LEFT, 11), heading, 26)
    query = case['query']
    draw_text(draw, (LEFT, 48), query[:140] + ('…' if len(query) > 140 else ''), 19)
    draw_text(draw, (LEFT, 77),
              f"原 G: {case['old_source']} / {case['old_count']} 候选    "
              f"G′: {case['new_source']} / {case['new_count']} 候选    "
              f"阶段: {case['change_stage']}    绿虚线=GT，底排固定放大 GT 附近", 19, '#b7c8dd')
    values = [('A', case['a'], case['a_iou']),
              ('G', case['g'], case['g_iou']),
              ("G′", case['new'], case['new_iou']),
              ('GT', case['gt'], 1.0)]
    zoom = gt_crop(case['gt'])
    for index, (name, box, iou) in enumerate(values):
        x = LEFT + index*(PANEL_W+PANEL_GAP)
        panel(canvas, image, x, 112, name, box, case['gt'], (0, 0, 1, 1), False, iou)
        panel(canvas, image, x, 458, name, box, case['gt'], zoom, True, iou)
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=91, optimize=True)


def classify(old_iou, new_iou):
    if old_iou < .5 <= new_iou:
        return 'improved'
    if new_iou < .5 <= old_iou:
        return 'worsened'
    return 'still_correct' if old_iou >= .5 else 'still_wrong'


def make_index(cases, counts, missing, output_dir, compared_count):
    sections = []
    for group, label in GROUPS:
        cards = []
        for case in cases:
            if case['group'] != group:
                continue
            cards.append(f'''<article class="card" id="q-{escape(case['id'])}">
  <div class="line"><b>{escape(case['id'])}</b><span>{escape(case['query'])}</span></div>
  <div class="metrics">A {case['a_iou']:.3f} · G {case['g_iou']:.3f} → G′ {case['new_iou']:.3f} ·
  候选 {case['old_count']} → {case['new_count']} · {escape(case['change_stage'])}</div>
  <a href="cases/{case['id']}.jpg" target="_blank"><img loading="lazy" src="cases/{case['id']}.jpg" alt="A, G, G′, GT full image and GT zoom for {case['id']}"></a>
</article>''')
        sections.append(f'<section id="{group}"><h2>{label} <small>{counts[group]} 题</small></h2>'
                        + ''.join(cards) + '</section>')
    nav = ' · '.join(f'<a href="#{key}">{label} {counts[key]}</a>' for key, label in GROUPS)
    html = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>G 与 G′ 变化题 · {len(cases)} 题</title>
<style>body{{margin:0;background:#0e1622;color:#e9f0f9;font-family:system-ui,"Microsoft YaHei",sans-serif}}
header{{padding:27px max(3vw,20px);background:#172536;position:sticky;top:0;z-index:2;box-shadow:0 4px 24px #0008}}
h1{{font-size:27px;margin:0 0 10px}}p{{margin:6px 0;color:#b8cadc}}a{{color:#8bd2ff}}
main{{max-width:1780px;margin:auto;padding:20px}}section{{margin-bottom:50px;scroll-margin-top:150px}}
h2{{border-left:5px solid #60cfff;padding-left:12px}}small{{color:#a4bbd0;font-size:16px}}
.card{{background:#19283a;border:1px solid #30465e;border-radius:12px;padding:17px;margin:20px 0}}
.line{{display:flex;gap:16px;align-items:baseline;flex-wrap:wrap;font-size:18px}}.line b{{color:#fff}}
.metrics{{color:#bdd0df;margin:9px 0}}img{{width:100%;height:auto;border:1px solid #3a526a;border-radius:6px}}
nav{{margin-top:13px}}nav a{{margin-right:10px}}.note{{color:#ffe2a3}}</style></head><body>
<header><h1>原 G → G′：仅展示输出框改变的 {len(cases)} 题</h1>
<p>共对比 {compared_count} 道有答案的序数题；G′ 是新候选日志按 G 规则所得结果。A、原 G、G′ 并排。绿虚线是 GT，最右侧专门显示 GT；底排固定放大 GT 附近，预测框若在视野外会标明。</p>
<p class="note">GT 与预测框均为原图归一化坐标。IoU≥0.5 计正确。原图解码失败 {missing} 张。</p>
<nav>{nav}</nav></header><main>{''.join(sections)}</main></body></html>'''
    (output_dir / 'index.html').write_text(html, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--old-eval-dir', type=Path, required=True,
                        help='Directory with original G queries_g.json and per_query.csv')
    parser.add_argument('--new-eval-dir', type=Path, required=True,
                        help='Directory with G-prime queries_g.json and per_query.csv')
    parser.add_argument('--middle-eval-dir', type=Path,
                        help='Optional intermediate G evaluation, to label which stage changed the box')
    parser.add_argument('--references', type=Path,
                        default=PROJECT / 'datasets/ordinal_subset/annotations.json')
    parser.add_argument('--baseline-predictions', type=Path,
                        default=PROJECT / 'outputs/rgb_all/queries_rgb.json')
    parser.add_argument('--data-root', type=Path,
                        default=PROJECT / 'datasets/reference_subset')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    gt = load_json(args.references)
    baseline = load_json(args.baseline_predictions)
    old_predictions = load_json(args.old_eval_dir / 'queries_g.json')
    new_predictions = load_json(args.new_eval_dir / 'queries_g.json')
    middle_predictions = (load_json(args.middle_eval_dir / 'queries_g.json')
                          if args.middle_eval_dir else None)
    old_details = load_csv(args.old_eval_dir / 'per_query.csv')
    new_details = load_csv(args.new_eval_dir / 'per_query.csv')
    missing_ids = set(new_predictions) - (set(gt) & set(baseline) & set(old_predictions)
                                          & set(old_details) & set(new_details))
    if missing_ids:
        raise ValueError(f'Missing reference, A, original G, or details for {sorted(missing_ids)}')
    if middle_predictions is not None and set(new_predictions) - set(middle_predictions):
        raise ValueError('Intermediate G evaluation is missing requested IDs')
    cases = []
    for key in new_predictions:
        g_box, new_box = old_predictions[key]['bbox'], new_predictions[key]['bbox']
        if g_box == new_box:
            continue
        gt_box, a_box = gt[key]['bbox'], baseline[key]['bbox']
        old_iou, new_iou = box_iou(g_box, gt_box), box_iou(new_box, gt_box)
        if middle_predictions is None:
            change_stage = 'G → G′'
        else:
            middle_box = middle_predictions[key]['bbox']
            change_stage = ('两阶段均改变' if g_box != middle_box and middle_box != new_box
                            else '前一阶段' if g_box != middle_box else '后一阶段')
        group = classify(old_iou, new_iou)
        cases.append(dict(id=key, group=group,
                          group_label=dict(GROUPS)[group], query=gt[key]['query'],
                          visible=gt[key]['visible'], gt=gt_box, a=a_box,
                          g=g_box, new=new_box, a_iou=box_iou(a_box, gt_box),
                          g_iou=old_iou, new_iou=new_iou,
                          change_stage=change_stage,
                          old_source=old_details[key]['source'],
                          new_source=new_details[key]['source'],
                          old_count=int(old_details[key]['candidate_count']),
                          new_count=int(new_details[key]['candidate_count'])))
    cases.sort(key=lambda c:(dict((key, n) for n, (key, _) in enumerate(GROUPS))[c['group']],
                             c['id']))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'cases').mkdir(exist_ok=True)
    missing = 0
    for index, case in enumerate(cases, 1):
        try:
            image, source = load_image(case['visible'], args.data_root)
            case['image_source'] = str(source)
        except (FileNotFoundError, OSError):
            missing += 1
            raise
        render_case(case, image, args.output_dir / 'cases' / f"{case['id']}.jpg")
        print(f'{index}/{len(cases)} {case["id"]} {case["group"]}', flush=True)
    counts = Counter(case['group'] for case in cases)
    make_index(cases, counts, missing, args.output_dir, len(new_predictions))
    fields = ['id', 'group', 'query', 'visible', 'a_iou', 'g_iou', 'new_iou',
              'old_source', 'new_source', 'old_count', 'new_count',
              'change_stage', 'image_source']
    with (args.output_dir / 'manifest.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows({field: case[field] for field in fields} for case in cases)
    print(json.dumps(dict(total=len(cases), groups=dict(counts), missing_rgb=missing,
                          index=str(args.output_dir / 'index.html')), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
