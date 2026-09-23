#!/usr/bin/env python3
"""Select explicit horizontal ordinal queries from the labeled preliminary set."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from prepare_reference_subset import load_json, valid_box, write_json


DIRECTION = re.compile(r'\b(left\s+to\s+right|right\s+to\s+left)\b', re.I)
ORDER_PHRASE = re.compile(
    r'\b(?:from\s+(?:the\s+)?)?(?:left\s+to\s+right|right\s+to\s+left)\b', re.I)
VERTICAL = re.compile(r'\b(?:top\s+to\s+bottom|bottom\s+to\s+top)\b', re.I)
ORDINAL = re.compile(
    r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|'
    r'[1-9](?:st|nd|rd|th)|10th)\b', re.I)
ORDINAL_VALUES = {
    word: number for number, word in enumerate(
        ('first', 'second', 'third', 'fourth', 'fifth', 'sixth',
         'seventh', 'eighth', 'ninth', 'tenth'), 1)
}


def parse_ordinal(query):
    """Return (rank, direction, target) only for unambiguous horizontal ordinals."""
    if not isinstance(query, str):
        return None
    directions = list(DIRECTION.finditer(query))
    ordinals = list(ORDINAL.finditer(query))
    if len(directions) != 1 or len(ordinals) != 1 or VERTICAL.search(query):
        return None
    word = ordinals[0].group().lower()
    rank = ORDINAL_VALUES.get(word, int(re.match(r'\d+', word).group()) if word[0].isdigit() else 0)
    direction = 'left_to_right' if directions[0].group().lower().startswith('left') else 'right_to_left'
    # Keep visual qualifiers (color, material, location); remove only ordering syntax.
    target = ORDER_PHRASE.sub(' ', query)
    target = ORDINAL.sub(' ', target)
    target = re.sub(r'\bfrom\s+the\s+(?:left|right)\s*,?\s*when\s+counting\b',
                    ' ', target, flags=re.I)
    target = re.sub(r'[(),;:.]+', ' ', target)
    target = re.sub(r'\s+', ' ', target).strip()
    target = re.sub(r'^(?:(?:from|the|a|an)\s+)+', '', target, flags=re.I).strip()
    if not target:
        return None
    return rank, direction, target


def select(queries, annotations):
    if set(queries) != set(annotations):
        raise ValueError('Query and annotation IDs differ')
    selected_queries, selected_answers, parsed = {}, {}, {}
    skipped = Counter()
    for key, item in queries.items():
        reference = annotations[key]
        if not isinstance(item, dict) or not isinstance(reference, dict):
            raise ValueError(f'{key}: invalid query/reference row')
        for field in ('query', 'visible'):
            if item.get(field) != reference.get(field):
                raise ValueError(f'{key}: reference {field} mismatch')
        if not valid_box(reference.get('bbox')):
            raise ValueError(f'{key}: invalid reference bbox')
        value = parse_ordinal(item.get('query'))
        if value is None:
            skipped['not_clear_horizontal_ordinal'] += 1
            continue
        rank, direction, target = value
        selected_queries[key] = {field: val for field, val in item.items() if field != 'bbox'}
        selected_answers[key] = reference
        parsed[key] = dict(rank=rank, direction=direction, target=target)
    return selected_queries, selected_answers, parsed, skipped


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queries', type=Path, default=root / 'datasets/reference_subset/queries.json')
    parser.add_argument('--references', type=Path, default=root / 'datasets/reference_subset/annotations.json')
    parser.add_argument('--output-dir', type=Path, default=root / 'datasets/ordinal_subset')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    queries = load_json(args.queries)
    references = load_json(args.references)
    subset, answers, parsed, skipped = select(queries, references)
    if not subset:
        raise ValueError('No explicit horizontal ordinal queries found')
    summary = dict(source_queries=len(queries), selected_queries=len(subset),
                   unique_images=len({item['visible'] for item in subset.values()}),
                   directions=dict(Counter(v['direction'] for v in parsed.values())),
                   skipped=dict(skipped),
                   scope='one explicit ordinal and one horizontal direction; mixed vertical directions excluded',
                   queries_sha256=hashlib.sha256(args.queries.read_bytes()).hexdigest(),
                   references_sha256=hashlib.sha256(args.references.read_bytes()).hexdigest())
    if not args.check_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / 'queries.json', subset)
        write_json(args.output_dir / 'annotations.json', answers)
        write_json(args.output_dir / 'parsed_ordinals.json', parsed)
        write_json(args.output_dir / 'summary.json', summary)
        (args.output_dir / 'README.md').write_text(
            '# 初赛左右序数参考子集\n\n'
            f'从 {len(queries)} 条人工参考题筛出 {len(subset)} 条。'
            'queries.json 不含 bbox；annotations.json 含人工 bbox。\n'
            'parsed_ordinals.json 记录自动提取的序号、方向及目标短语，请人工检查。\n'
            '本目录不复制图片；推理时 --data-root 指向原 reference_subset 目录。\n',
            encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
