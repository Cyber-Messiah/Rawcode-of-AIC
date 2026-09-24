"""Decode LocateAnything box tokens into normalized, canonical xyxy boxes.

The four generated coordinates describe two corners. A model can emit those
corners in either order; ordering is canonicalized before geometry validation.
"""
from collections import Counter
import math
import re


BOX = re.compile(r'<box>(.*?)</box>', re.I | re.S)
NUMBER = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
TAGGED = re.compile(r'\s*' + r'\s*'.join(fr'<({NUMBER})>' for _ in range(4)) + r'\s*')
COORDINATES = r'\s*,\s*'.join(fr'({NUMBER})' for _ in range(4))
PLAIN = re.compile(r'\s*(?:\(' + COORDINATES + r'\)|\[' + COORDINATES +
                   r'\]|' + COORDINATES + r')\s*')


def valid_box(box):
    """Check a normalized xyxy box after coordinate order has been resolved."""
    return (isinstance(box, list) and len(box) == 4
            and all(type(value) in (int, float) and math.isfinite(value)
                    and 0 <= value <= 1 for value in box)
            and box[0] < box[2] and box[1] < box[3])


def parse_boxes(answer, image_size=None):
    """Return (canonical boxes, audit) without clipping or double scaling.

    LocateAnything's angle-bracket tokens use 0..1000 units. Plain 0..1
    coordinates are already normalized; other plain values through 1000 use
    model units. Larger plain coordinates are pixels only when image dimensions
    make that interpretation possible.
    """
    boxes = []
    audit = dict(box_tags=0, accepted=0, reversed_corners=0,
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
        groups = [value for value in found.groups() if value is not None]
        numbers = [float(value) for value in groups]
        if any(not math.isfinite(value) or value < 0 for value in numbers):
            audit['rejected']['nonfinite_or_negative'] += 1
            continue
        if tagged:
            unit = 'model_1000'
            if any(value > 1000 for value in numbers):
                audit['rejected']['tagged_out_of_range'] += 1
                continue
        elif all(value <= 1 for value in numbers):
            unit = 'normalized'
        elif all(value <= 1000 for value in numbers):
            unit = 'model_1000'
        elif (image_size and image_size[0] > 0 and image_size[1] > 0
              and numbers[0] <= image_size[0] and numbers[2] <= image_size[0]
              and numbers[1] <= image_size[1] and numbers[3] <= image_size[1]):
            unit = 'pixel'
        else:
            audit['rejected']['out_of_range_or_ambiguous_pixels'] += 1
            continue
        if unit == 'model_1000':
            values = [value / 1000 for value in numbers]
        elif unit == 'pixel':
            values = [value / image_size[index % 2] for index, value in enumerate(numbers)]
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
