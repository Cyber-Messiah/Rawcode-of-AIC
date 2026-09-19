#!/usr/bin/env python3
"""Convert competition-style grounding JSON into LocateAnything SFT JSONL."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


MODE_TO_FIELDS = {
    "rgb": ("visible",),
    "rgb_t": ("visible", "infrared"),
    "rgb_d": ("visible", "depth"),
    "rgb_d_t": ("visible", "infrared", "depth"),
}

MODE_TO_PROMPT = {
    "rgb": "Locate a single instance that matches the following description: {query}",
    "rgb_t": "Locate a single instance that matches the following description: {query}",
    "rgb_d": "Locate a single instance that matches the following description: {query}",
    "rgb_d_t": "Locate a single instance that matches the following description: {query}",
}

FIELD_TO_MODALITY = {
    "infrared": "thermal",
    "depth": "depth",
}


def load_competition_json(path: Path) -> list[tuple[str, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        return list(data.items())
    if isinstance(data, list):
        return [(str(i), item) for i, item in enumerate(data)]
    raise TypeError(f"Unsupported JSON root type: {type(data).__name__}")


def bbox_to_locany_tokens(bbox: list[float]) -> tuple[int, int, int, int]:
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 numbers, got {bbox}")
    vals = [max(0, min(1000, int(round(float(v) * 1000)))) for v in bbox]
    x1, y1, x2, y2 = vals
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid bbox after conversion: {bbox} -> {vals}")
    return x1, y1, x2, y2


def validate_paths(root: Path, rel_paths: list[str], require_exists: bool) -> bool:
    if not require_exists:
        return True
    return all((root / rel).is_file() for rel in rel_paths)


def convert_item(
    item_id: str,
    item: dict[str, Any],
    mode: str,
    root: Path,
    require_exists: bool,
    legacy_multi_image: bool,
) -> dict[str, Any] | None:
    query = str(item.get("query", "")).strip()
    if not query:
        return None

    fields = MODE_TO_FIELDS[mode]
    images: list[str] = []
    for field in fields:
        path = item.get(field)
        if not path:
            return None
        images.append(str(path))

    if not validate_paths(root, images, require_exists):
        return None

    x1, y1, x2, y2 = bbox_to_locany_tokens(item["bbox"])
    prompt = MODE_TO_PROMPT[mode].format(query=query)
    answer = f"<ref>{query}</ref><box>({x1},{y1},{x2},{y2})</box>"

    result = {
        "id": item_id,
        "image": images[0] if not legacy_multi_image or len(images) == 1 else images,
        "conversations": [
            {"from": "human", "value": prompt},
            {"from": "gpt", "value": answer},
        ],
    }
    if not legacy_multi_image and len(images) > 1:
        result["auxiliary_images"] = [
            {"path": path, "modality": FIELD_TO_MODALITY[field]}
            for field, path in zip(fields[1:], images[1:])
        ]
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=sorted(MODE_TO_FIELDS))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--no-path-check", action="store_true")
    parser.add_argument(
        "--legacy-multi-image",
        action="store_true",
        help="Put every modality in the image list instead of feature fusion metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_competition_json(args.input_json)
    if args.shuffle:
        random.Random(args.seed).shuffle(rows)

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with args.output_jsonl.open("w", encoding="utf-8") as out:
        for item_id, item in rows:
            if args.limit and written >= args.limit:
                break
            try:
                converted = convert_item(
                    item_id,
                    item,
                    args.mode,
                    args.root,
                    require_exists=not args.no_path_check,
                    legacy_multi_image=args.legacy_multi_image,
                )
            except Exception:
                converted = None
            if converted is None:
                skipped += 1
                continue
            out.write(json.dumps(converted, ensure_ascii=False) + "\n")
            written += 1

    print(
        json.dumps(
            {
                "input": str(args.input_json),
                "root": str(args.root),
                "output": str(args.output_jsonl),
                "mode": args.mode,
                "written": written,
                "skipped": skipped,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()