#!/usr/bin/env python3
"""Evaluate RGB-anchored LocateAnything fusion checkpoints."""

from __future__ import annotations

import argparse
import json
import random
import re
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from PIL import Image, ImageDraw
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer


MODE_TO_AUX = {
    "rgb": (),
    "rgb_t": (("infrared", 1),),
    "rgb_d": (("depth", 2),),
    "rgb_d_t": (("infrared", 1), ("depth", 2)),
}
BOX_PATTERNS = (
    re.compile(
        r"<box>\s*\(?\s*(\d+)\s*[,<>]\s*(\d+)\s*[,<>]\s*"
        r"(\d+)\s*[,<>]\s*(\d+)\s*\)?\s*</box>"
    ),
    re.compile(r"<box>\s*<(\d+)><(\d+)><(\d+)><(\d+)>\s*</box>"),
)


class RGBTLocalCrossAttentionFusion(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            bottleneck_dim: int,
            window_size: int,
            max_residual_scale: float):
        super().__init__()
        if window_size < 1 or window_size % 2 == 0:
            raise ValueError("fusion_window_size must be a positive odd integer.")
        self.window_size = window_size
        self.max_residual_scale = max_residual_scale
        self.rgb_norm = nn.LayerNorm(feature_dim)
        self.thermal_norm = nn.LayerNorm(feature_dim)
        self.query = nn.Linear(feature_dim, bottleneck_dim)
        self.key = nn.Linear(feature_dim, bottleneck_dim)
        self.value = nn.Linear(feature_dim, bottleneck_dim)
        self.context_norm = nn.LayerNorm(bottleneck_dim)
        self.output = nn.Linear(bottleneck_dim, feature_dim)
        gate_hidden = max(32, bottleneck_dim // 2)
        self.token_gate = nn.Sequential(
            nn.Linear(bottleneck_dim * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        self.global_gate = nn.Sequential(
            nn.Linear(bottleneck_dim * 2, gate_hidden),
            nn.GELU(),
            nn.Linear(gate_hidden, 1),
        )
        self.relative_position_bias = nn.Parameter(
            torch.zeros(window_size * window_size)
        )
        self.attention_scale = bottleneck_dim ** -0.5

    def _local_neighborhoods(
            self,
            tokens: torch.Tensor,
            height: int,
            width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        channels = tokens.shape[-1]
        feature_map = tokens.reshape(height, width, channels).permute(2, 0, 1)
        neighborhoods = F.unfold(
            feature_map.unsqueeze(0),
            kernel_size=self.window_size,
            padding=self.window_size // 2,
        )
        neighborhoods = (
            neighborhoods.reshape(
                1, channels, self.window_size * self.window_size, height * width
            )
            .permute(0, 3, 2, 1)
            .squeeze(0)
        )
        valid = F.unfold(
            torch.ones(
                (1, 1, height, width),
                device=tokens.device,
                dtype=tokens.dtype,
            ),
            kernel_size=self.window_size,
            padding=self.window_size // 2,
        )
        return neighborhoods, valid.squeeze(0).transpose(0, 1).bool()

    def forward(
            self,
            rgb_tokens: torch.Tensor,
            thermal_tokens: torch.Tensor,
            grid_hw: tuple[int, int],
    ) -> torch.Tensor:
        height, width = grid_hw
        rgb_norm = self.rgb_norm(rgb_tokens)
        thermal_norm = self.thermal_norm(thermal_tokens)
        query = self.query(rgb_norm)
        key = self.key(thermal_norm)
        value = self.value(thermal_norm)
        local_key, valid = self._local_neighborhoods(key, height, width)
        local_value, _ = self._local_neighborhoods(value, height, width)
        logits = (
            (query.unsqueeze(1) * local_key).sum(dim=-1) * self.attention_scale
        )
        logits = logits + self.relative_position_bias
        logits = logits.masked_fill(~valid, -1e4)
        attention = torch.softmax(logits.float(), dim=-1).to(local_value.dtype)
        context = (attention.unsqueeze(-1) * local_value).sum(dim=1)
        context = self.context_norm(context)
        residual = self.output(context)
        gate_input = torch.cat((query, context), dim=-1)
        token_gate = torch.sigmoid(self.token_gate(gate_input))
        global_gate = torch.sigmoid(
            self.global_gate(gate_input.mean(dim=0, keepdim=True))
        )
        rgb_rms = (
            rgb_tokens.float().square().mean(dim=-1, keepdim=True).sqrt()
            .clamp_min(1e-6)
            .to(dtype=rgb_tokens.dtype)
        )
        bounded_residual = torch.tanh(residual) * rgb_rms
        return (
            self.max_residual_scale
            * token_gate
            * global_gate
            * bounded_residual
        )


def attach_modality_fusion(model, checkpoint: Path | None) -> None:
    feature_dim = int(model.config.vision_config.hidden_size) * 4
    bottleneck_dim = 256
    window_size = 5
    max_residual_scale = 0.1
    if checkpoint:
        config_path = checkpoint / "config.json" if checkpoint.is_dir() else None
        if config_path and config_path.exists():
            with config_path.open("r", encoding="utf-8") as file:
                fusion_config = json.load(file)
            bottleneck_dim = int(
                fusion_config.get("fusion_bottleneck_dim", bottleneck_dim)
            )
            window_size = int(
                fusion_config.get("fusion_window_size", window_size)
            )
            max_residual_scale = float(
                fusion_config.get(
                    "fusion_max_residual_scale", max_residual_scale
                )
            )
    model.modality_adapters = nn.ModuleDict({
        "1": RGBTLocalCrossAttentionFusion(
            feature_dim,
            bottleneck_dim,
            window_size,
            max_residual_scale,
        ),
    })
    for adapter in model.modality_adapters.values():
        adapter.rgb_norm.reset_parameters()
        adapter.thermal_norm.reset_parameters()
        adapter.context_norm.reset_parameters()
        for projection in (adapter.query, adapter.key, adapter.value):
            projection.reset_parameters()
        nn.init.zeros_(adapter.output.weight)
        nn.init.zeros_(adapter.output.bias)
        adapter.token_gate[0].reset_parameters()
        nn.init.zeros_(adapter.token_gate[-1].weight)
        nn.init.constant_(adapter.token_gate[-1].bias, -2.0)
        adapter.global_gate[0].reset_parameters()
        nn.init.zeros_(adapter.global_gate[-1].weight)
        nn.init.constant_(adapter.global_gate[-1].bias, -2.0)
        nn.init.zeros_(adapter.relative_position_bias)

    if checkpoint:
        checkpoint_path = checkpoint
        if checkpoint_path.is_dir():
            checkpoint_path = checkpoint_path / "trainable_state.pt"
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected checkpoint keys: {incompatible.unexpected_keys}")

    base_extract_feature = model.extract_feature

    def fused_extract_feature(self, pixel_values, image_grid_hws):
        embeds = base_extract_feature(pixel_values, image_grid_hws)
        modality_ids = getattr(self, "_fusion_modality_ids", [0])
        if len(modality_ids) != len(embeds):
            raise ValueError(
                f"modality/image mismatch: {len(modality_ids)} ids, {len(embeds)} images"
            )
        rgb_tokens = embeds[0]
        rgb_grid = image_grid_hws[0]
        merge_h, merge_w = self.config.vision_config.merge_kernel_size
        target_h = int(rgb_grid[0]) // int(merge_h)
        target_w = int(rgb_grid[1]) // int(merge_w)
        fused = rgb_tokens

        for aux_tokens, modality_id, aux_grid in zip(
                embeds[1:], modality_ids[1:], image_grid_hws[1:]):
            source_h = int(aux_grid[0]) // int(merge_h)
            source_w = int(aux_grid[1]) // int(merge_w)
            if (source_h, source_w) != (target_h, target_w):
                dtype = aux_tokens.dtype
                feature_map = (
                    aux_tokens.reshape(source_h, source_w, -1)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                )
                feature_map = F.interpolate(
                    feature_map.float(),
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=False,
                ).to(dtype=dtype)
                aux_tokens = (
                    feature_map.squeeze(0)
                    .permute(1, 2, 0)
                    .reshape(target_h * target_w, -1)
                )
            if int(modality_id) != 1:
                raise ValueError("This checkpoint supports RGB-T fusion only.")
            fused = fused + self.modality_adapters[str(modality_id)](
                rgb_tokens,
                aux_tokens,
                (target_h, target_w),
            )
        return [fused]

    model.extract_feature = types.MethodType(fused_extract_feature, model)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model-path", default="nvidia/LocateAnything-3B")
    parser.add_argument("--fusion-checkpoint", type=Path)
    parser.add_argument("--mode", choices=sorted(MODE_TO_AUX), default="rgb_t")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--visualize-count", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_annotations(path: Path) -> list[tuple[str, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if isinstance(data, dict):
        return list(data.items())
    return [(str(index), item) for index, item in enumerate(data)]


def parse_box(answer: str) -> list[float] | None:
    for pattern in BOX_PATTERNS:
        match = pattern.search(answer)
        if match:
            values = [max(0, min(1000, int(value))) / 1000 for value in match.groups()]
            if values[2] > values[0] and values[3] > values[1]:
                return values
    return None


def box_iou(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0 else 0.0


def build_messages(image: Image.Image, query: str) -> list[dict[str, Any]]:
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {
                "type": "text",
                "text": (
                    "Locate a single instance that matches the following "
                    f"description: {query}"
                ),
            },
        ],
    }]


def answer_to_text(answer: Any, tokenizer) -> str:
    if isinstance(answer, torch.Tensor):
        return tokenizer.batch_decode(answer, skip_special_tokens=False)[0]
    if isinstance(answer, str):
        return answer
    if isinstance(answer, tuple) and answer:
        return answer_to_text(answer[0], tokenizer)
    if isinstance(answer, list) and answer:
        return answer_to_text(answer[0], tokenizer)
    return str(answer)


def draw_result(
        image: Image.Image,
        query: str,
        ground_truth: list[float],
        prediction: list[float] | None,
        iou: float,
) -> Image.Image:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size

    def pixel_box(box: list[float]) -> tuple[int, int, int, int]:
        return (
            round(box[0] * width),
            round(box[1] * height),
            round(box[2] * width),
            round(box[3] * height),
        )

    draw.rectangle(pixel_box(ground_truth), outline=(40, 220, 80), width=4)
    if prediction is not None:
        draw.rectangle(pixel_box(prediction), outline=(240, 55, 55), width=4)
    label = f"GT green | Pred red | IoU {iou:.3f}\n{query}"
    text_box = draw.multiline_textbbox((0, 0), label)
    text_height = text_box[3] - text_box[1] + 12
    draw.rectangle((0, 0, width, text_height), fill=(0, 0, 0))
    draw.multiline_text((6, 5), label, fill=(255, 255, 255))
    return canvas


def make_contact_sheet(images: list[Image.Image], output_path: Path) -> None:
    if not images:
        return
    columns = 4
    thumb_width, thumb_height = 420, 280
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), "white")
    for index, image in enumerate(images):
        thumbnail = image.copy()
        thumbnail.thumbnail((thumb_width, thumb_height))
        x = (index % columns) * thumb_width + (thumb_width - thumbnail.width) // 2
        y = (index // columns) * thumb_height + (thumb_height - thumbnail.height) // 2
        sheet.paste(thumbnail, (x, y))
    sheet.save(output_path, quality=92)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    visual_dir = args.output_dir / "visualizations"
    visual_dir.mkdir(exist_ok=True)

    model_config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    model_config._attn_implementation = "sdpa"
    model_config.text_config._attn_implementation = "sdpa"
    model_config.text_config._attn_implementation_internal = "sdpa"
    model_config.vision_config._attn_implementation = "flash_attention_2"
    model = AutoModel.from_pretrained(
        args.model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.language_model.model._attn_implementation = "sdpa"
    attach_modality_fusion(model, args.fusion_checkpoint)
    model = model.to(device="cuda", dtype=torch.bfloat16).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    processor.tokenizer = tokenizer

    rows = load_annotations(args.annotation)
    if args.limit > 0:
        rows = rows[:args.limit]
    results = []
    visualizations = []
    valid_ious = []
    correct = 0

    for index, (sample_id, item) in enumerate(rows):
        rgb_path = args.data_root / item["visible"]
        with Image.open(rgb_path) as file:
            rgb = file.convert("RGB")
        query = str(item["query"])
        messages = build_messages(rgb, query)
        text = processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = processor.process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=images,
            videos=videos,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"]
        grid_hws = inputs["image_grid_hws"]
        modality_ids = [0]
        for field, modality_id in MODE_TO_AUX[args.mode]:
            path = item.get(field)
            if not path:
                continue
            with Image.open(args.data_root / path) as file:
                auxiliary = file.convert("RGB")
            auxiliary_inputs = processor.image_processor(
                images=[auxiliary], return_tensors="pt"
            )
            pixel_values = torch.cat(
                (pixel_values, auxiliary_inputs["pixel_values"]), dim=0
            )
            grid_hws = np.concatenate(
                (grid_hws, auxiliary_inputs["image_grid_hws"]), axis=0
            )
            modality_ids.append(modality_id)

        with torch.no_grad():
            model._fusion_modality_ids = modality_ids
            generated = model.generate(
                pixel_values=pixel_values.cuda().to(torch.bfloat16),
                input_ids=inputs["input_ids"].cuda(),
                attention_mask=inputs["attention_mask"].cuda(),
                image_grid_hws=torch.as_tensor(grid_hws, device="cuda"),
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
                generation_mode="hybrid",
                tokenizer=tokenizer,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )

        answer = answer_to_text(generated, tokenizer)
        prediction = parse_box(answer)
        ground_truth = [float(value) for value in item["bbox"]]
        iou = box_iou(prediction, ground_truth) if prediction else 0.0
        if prediction is not None:
            valid_ious.append(iou)
        correct += int(iou >= 0.5)
        result = {
            "id": sample_id,
            "query": query,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "iou": round(iou, 6),
            "acc_at_0_5": iou >= 0.5,
            "answer": answer,
        }
        results.append(result)

        if index < args.visualize_count:
            rendered = draw_result(rgb, query, ground_truth, prediction, iou)
            rendered.save(visual_dir / f"{index:03d}_{sample_id}.jpg", quality=92)
            visualizations.append(rendered)
        print(
            f"[{index + 1}/{len(rows)}] {sample_id} "
            f"IoU={iou:.3f} valid={prediction is not None}",
            flush=True,
        )

    with (args.output_dir / "results.jsonl").open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    total = len(results)
    metrics = {
        "mode": args.mode,
        "seed": args.seed,
        "samples": total,
        "valid_predictions": len(valid_ious),
        "valid_rate": len(valid_ious) / total if total else 0.0,
        "mean_iou": sum(result["iou"] for result in results) / total if total else 0.0,
        "acc_at_0_5": correct / total if total else 0.0,
    }
    with (args.output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    make_contact_sheet(visualizations, args.output_dir / "contact_sheet.jpg")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
