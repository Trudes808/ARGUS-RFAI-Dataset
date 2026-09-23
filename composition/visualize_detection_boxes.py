#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ImageRecord:
    id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class BoxRecord:
    id: int
    image_id: int
    category_id: int
    bbox: tuple[float, float, float, float]
    score: float | None
    source: str


@dataclass(frozen=True)
class MatchRecord:
    image_id: int
    category_id: int
    gt_id: int | None
    pred_id: int | None
    iou: float
    result: str


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def load_coco(path: Path, *, source: str) -> tuple[dict[int, ImageRecord], list[BoxRecord], dict[int, str], dict[str, Any]]:
    data = load_json(path)
    images = {
        int(item["id"]): ImageRecord(
            id=int(item["id"]),
            file_name=str(item["file_name"]),
            width=int(item["width"]),
            height=int(item["height"]),
        )
        for item in data.get("images", [])
    }
    categories = {int(item["id"]): str(item["name"]) for item in data.get("categories", [])}
    boxes = [
        BoxRecord(
            id=int(item.get("id", index + 1)),
            image_id=int(item["image_id"]),
            category_id=int(item["category_id"]),
            bbox=tuple(float(v) for v in item["bbox"]),
            score=None if item.get("score") is None else float(item["score"]),
            source=source,
        )
        for index, item in enumerate(data.get("annotations", []))
    ]
    return images, boxes, categories, data


def load_yolo_predictions(
    label_dir: Path,
    images: dict[int, ImageRecord],
    category_ids: list[int],
    *,
    default_score: float,
) -> list[BoxRecord]:
    predictions: list[BoxRecord] = []
    next_id = 1
    for image in images.values():
        label_path = label_dir / image.file_name.replace(".png", ".txt")
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) < 5:
                raise ValueError(f"Bad YOLO line in {label_path}: {line!r}")
            class_index = int(float(parts[0]))
            if class_index < 0 or class_index >= len(category_ids):
                raise ValueError(f"YOLO class index {class_index} is outside known category range")
            x_center = float(parts[1]) * image.width
            y_center = float(parts[2]) * image.height
            width = float(parts[3]) * image.width
            height = float(parts[4]) * image.height
            score = float(parts[5]) if len(parts) >= 6 else default_score
            predictions.append(
                BoxRecord(
                    id=next_id,
                    image_id=image.id,
                    category_id=category_ids[class_index],
                    bbox=(x_center - 0.5 * width, y_center - 0.5 * height, width, height),
                    score=score,
                    source="pred",
                )
            )
            next_id += 1
    return predictions


def clip_bbox(bbox: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float] | None:
    x, y, w, h = bbox
    x0 = max(0.0, x)
    y0 = max(0.0, y)
    x1 = min(float(width), x + w)
    y1 = min(float(height), y + h)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1 - x0, y1 - y0


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax1, ay1 = ax + aw, ay + ah
    bx1, by1 = bx + bw, by + bh
    ix0 = max(ax, bx)
    iy0 = max(ay, by)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def group_by_image(boxes: list[BoxRecord]) -> dict[int, list[BoxRecord]]:
    grouped: dict[int, list[BoxRecord]] = {}
    for box in boxes:
        grouped.setdefault(box.image_id, []).append(box)
    return grouped


def match_boxes(
    gt_boxes: list[BoxRecord],
    pred_boxes: list[BoxRecord],
    *,
    iou_threshold: float,
) -> list[MatchRecord]:
    matches: list[MatchRecord] = []
    gt_by_image = group_by_image(gt_boxes)
    pred_by_image = group_by_image(pred_boxes)
    image_ids = sorted(set(gt_by_image) | set(pred_by_image))

    for image_id in image_ids:
        local_gt = gt_by_image.get(image_id, [])
        local_pred = sorted(
            pred_by_image.get(image_id, []),
            key=lambda item: item.score if item.score is not None else 1.0,
            reverse=True,
        )
        unmatched_gt = set(range(len(local_gt)))
        for pred in local_pred:
            best_index = None
            best_iou = 0.0
            for gt_index in unmatched_gt:
                gt = local_gt[gt_index]
                if gt.category_id != pred.category_id:
                    continue
                iou = bbox_iou(gt.bbox, pred.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_index = gt_index
            if best_index is not None and best_iou >= iou_threshold:
                gt = local_gt[best_index]
                unmatched_gt.remove(best_index)
                matches.append(
                    MatchRecord(
                        image_id=image_id,
                        category_id=pred.category_id,
                        gt_id=gt.id,
                        pred_id=pred.id,
                        iou=best_iou,
                        result="tp",
                    )
                )
            else:
                matches.append(
                    MatchRecord(
                        image_id=image_id,
                        category_id=pred.category_id,
                        gt_id=None,
                        pred_id=pred.id,
                        iou=best_iou,
                        result="fp",
                    )
                )

        for gt_index in sorted(unmatched_gt):
            gt = local_gt[gt_index]
            matches.append(
                MatchRecord(
                    image_id=image_id,
                    category_id=gt.category_id,
                    gt_id=gt.id,
                    pred_id=None,
                    iou=0.0,
                    result="fn",
                )
            )
    return matches


def summarize_matches(matches: list[MatchRecord], categories: dict[int, str]) -> dict[str, Any]:
    by_category: dict[int, dict[str, float]] = {}
    for match in matches:
        stats = by_category.setdefault(
            match.category_id,
            {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0},
        )
        stats[match.result] += 1
        if match.result == "tp":
            stats["iou_sum"] += match.iou

    categories_out: dict[str, Any] = {}
    totals = {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0}
    for category_id, stats in sorted(by_category.items()):
        tp = int(stats["tp"])
        fp = int(stats["fp"])
        fn = int(stats["fn"])
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        mean_iou = stats["iou_sum"] / tp if tp else 0.0
        categories_out[str(category_id)] = {
            "name": categories.get(category_id, str(category_id)),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "mean_iou": mean_iou,
        }
        totals["tp"] += tp
        totals["fp"] += fp
        totals["fn"] += fn
        totals["iou_sum"] += stats["iou_sum"]

    tp = int(totals["tp"])
    fp = int(totals["fp"])
    fn = int(totals["fn"])
    return {
        "overall": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": tp / (tp + fp) if (tp + fp) else 0.0,
            "recall": tp / (tp + fn) if (tp + fn) else 0.0,
            "mean_iou": totals["iou_sum"] / tp if tp else 0.0,
        },
        "categories": categories_out,
    }


def generate_random_predictions(
    gt_data: dict[str, Any],
    *,
    seed: int,
    drop_probability: float,
    jitter_fraction: float,
    size_jitter_fraction: float,
    false_positive_rate: float,
    score_min: float,
    score_max: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    images = {int(item["id"]): item for item in gt_data.get("images", [])}
    categories = gt_data.get("categories", [])
    category_ids = [int(item["id"]) for item in categories]
    annotations: list[dict[str, Any]] = []
    next_id = 1

    for ann in gt_data.get("annotations", []):
        image = images[int(ann["image_id"])]
        if rng.random() < drop_probability:
            continue
        x, y, w, h = (float(v) for v in ann["bbox"])
        cx = x + 0.5 * w + rng.normal(0.0, jitter_fraction * max(w, 1.0))
        cy = y + 0.5 * h + rng.normal(0.0, jitter_fraction * max(h, 1.0))
        scale_w = max(0.05, 1.0 + rng.normal(0.0, size_jitter_fraction))
        scale_h = max(0.05, 1.0 + rng.normal(0.0, size_jitter_fraction))
        out_w = max(1.0, w * scale_w)
        out_h = max(1.0, h * scale_h)
        clipped = clip_bbox(
            (cx - 0.5 * out_w, cy - 0.5 * out_h, out_w, out_h),
            int(image["width"]),
            int(image["height"]),
        )
        if clipped is None:
            continue
        annotations.append(
            {
                "id": next_id,
                "image_id": int(ann["image_id"]),
                "category_id": int(ann["category_id"]),
                "bbox": list(clipped),
                "area": clipped[2] * clipped[3],
                "iscrowd": 0,
                "score": float(rng.uniform(score_min, score_max)),
                "random:source_gt_id": int(ann.get("id", -1)),
            }
        )
        next_id += 1

    for image in gt_data.get("images", []):
        false_count = int(rng.poisson(false_positive_rate))
        for _ in range(false_count):
            width = float(rng.uniform(8.0, max(9.0, 0.35 * float(image["width"]))))
            height = float(rng.uniform(8.0, max(9.0, 0.25 * float(image["height"]))))
            x = float(rng.uniform(0.0, max(1.0, float(image["width"]) - width)))
            y = float(rng.uniform(0.0, max(1.0, float(image["height"]) - height)))
            category_id = int(rng.choice(category_ids)) if category_ids else 1
            annotations.append(
                {
                    "id": next_id,
                    "image_id": int(image["id"]),
                    "category_id": category_id,
                    "bbox": [x, y, width, height],
                    "area": width * height,
                    "iscrowd": 0,
                    "score": float(rng.uniform(score_min, score_max)),
                    "random:false_positive": True,
                }
            )
            next_id += 1

    return {
        "info": {
            "description": "Random jittered prediction boxes generated for visualization testing",
            "random": {
                "seed": seed,
                "drop_probability": drop_probability,
                "jitter_fraction": jitter_fraction,
                "size_jitter_fraction": size_jitter_fraction,
                "false_positive_rate": false_positive_rate,
            },
        },
        "images": gt_data.get("images", []),
        "annotations": annotations,
        "categories": categories,
    }


def write_matches_csv(path: Path, matches: list[MatchRecord], categories: dict[int, str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_id", "category_id", "category_name", "gt_id", "pred_id", "iou", "result"],
        )
        writer.writeheader()
        for match in matches:
            writer.writerow(
                {
                    "image_id": match.image_id,
                    "category_id": match.category_id,
                    "category_name": categories.get(match.category_id, str(match.category_id)),
                    "gt_id": "" if match.gt_id is None else match.gt_id,
                    "pred_id": "" if match.pred_id is None else match.pred_id,
                    "iou": match.iou,
                    "result": match.result,
                }
            )


def image_path_for(image: ImageRecord, image_root: Path | None) -> Path | None:
    if image_root is None:
        return None
    direct = image_root / image.file_name
    if direct.exists():
        return direct
    nested = list(image_root.rglob(image.file_name))
    if nested:
        return nested[0]
    return None


def select_images(
    images: dict[int, ImageRecord],
    gt_by_image: dict[int, list[BoxRecord]],
    pred_by_image: dict[int, list[BoxRecord]],
    matches: list[MatchRecord],
    *,
    max_images: int,
    mode: str,
    seed: int,
) -> list[ImageRecord]:
    image_ids = [image_id for image_id in sorted(images) if gt_by_image.get(image_id) or pred_by_image.get(image_id)]
    if mode == "random":
        rng = np.random.default_rng(seed)
        rng.shuffle(image_ids)
    elif mode == "worst" and matches:
        by_image: dict[int, dict[str, float]] = {}
        for match in matches:
            stats = by_image.setdefault(match.image_id, {"tp": 0, "fp": 0, "fn": 0, "iou_sum": 0.0})
            stats[match.result] += 1
            if match.result == "tp":
                stats["iou_sum"] += match.iou
        image_ids.sort(
            key=lambda image_id: (
                by_image.get(image_id, {}).get("fn", 0) + by_image.get(image_id, {}).get("fp", 0),
                -by_image.get(image_id, {}).get("iou_sum", 0.0),
            ),
            reverse=True,
        )
    if max_images > 0:
        image_ids = image_ids[:max_images]
    return [images[image_id] for image_id in image_ids]


def render_overlays(
    *,
    outdir: Path,
    images: list[ImageRecord],
    gt_by_image: dict[int, list[BoxRecord]],
    pred_by_image: dict[int, list[BoxRecord]],
    matches: list[MatchRecord],
    categories: dict[int, str],
    image_root: Path | None,
    title_prefix: str,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    overlay_dir = outdir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()

    pred_status: dict[int, str] = {}
    gt_status: dict[int, str] = {}
    iou_by_pred: dict[int, float] = {}
    for match in matches:
        if match.pred_id is not None:
            pred_status[match.pred_id] = match.result
            iou_by_pred[match.pred_id] = match.iou
        if match.gt_id is not None:
            gt_status[match.gt_id] = match.result

    for image in images:
        bg_path = image_path_for(image, image_root)
        if bg_path is not None:
            canvas = Image.open(bg_path).convert("RGB")
            if canvas.size != (image.width, image.height):
                canvas = canvas.resize((image.width, image.height))
        else:
            canvas = Image.new("RGB", (image.width, image.height), (10, 10, 10))
        draw = ImageDraw.Draw(canvas)

        for gt in gt_by_image.get(image.id, []):
            status = gt_status.get(gt.id, "gt")
            color = (53, 255, 107) if status == "tp" else (255, 191, 60)
            x, y, w, h = gt.bbox
            draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
            label = categories.get(gt.category_id, str(gt.category_id))
            text = f"GT {label}"
            text_y = max(0.0, y - 12.0)
            bbox = draw.textbbox((x, text_y), text, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0))
            draw.text((x, text_y), text, fill=color, font=font)

        for pred in pred_by_image.get(image.id, []):
            status = pred_status.get(pred.id, "fp")
            color = (53, 167, 255) if status == "tp" else (255, 79, 94)
            x, y, w, h = pred.bbox
            # Dashed prediction rectangle.
            dash = 8
            gap = 5
            x0, y0, x1, y1 = x, y, x + w, y + h
            pos = x0
            while pos < x1:
                draw.line((pos, y0, min(pos + dash, x1), y0), fill=color, width=2)
                draw.line((pos, y1, min(pos + dash, x1), y1), fill=color, width=2)
                pos += dash + gap
            pos = y0
            while pos < y1:
                draw.line((x0, pos, x0, min(pos + dash, y1)), fill=color, width=2)
                draw.line((x1, pos, x1, min(pos + dash, y1)), fill=color, width=2)
                pos += dash + gap
            label = categories.get(pred.category_id, str(pred.category_id))
            score = "" if pred.score is None else f" {pred.score:.2f}"
            iou = iou_by_pred.get(pred.id)
            iou_text = "" if iou is None or status != "tp" else f" IoU {iou:.2f}"
            text = f"P {label}{score}{iou_text}"
            text_y = min(float(image.height) - 12.0, y + h + 2.0)
            bbox = draw.textbbox((x, text_y), text, font=font)
            draw.rectangle(bbox, fill=(0, 0, 0))
            draw.text((x, text_y), text, fill=color, font=font)

        title = f"{title_prefix}{image.file_name}"
        bbox = draw.textbbox((4, 4), title, font=font)
        draw.rectangle(bbox, fill=(0, 0, 0))
        draw.text((4, 4), title, fill=(230, 230, 230), font=font)
        out_path = overlay_dir / image.file_name.replace(".png", "_boxes.png")
        canvas.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize COCO/DINO spectrogram bounding boxes, optionally compare "
            "them to prediction boxes, and generate random prediction boxes for examples."
        )
    )
    parser.add_argument("coco", type=Path, help="Ground-truth COCO/DINO JSON.")
    pred = parser.add_mutually_exclusive_group()
    pred.add_argument("--pred-coco", type=Path, help="Prediction COCO JSON with bbox and optional score.")
    pred.add_argument("--pred-yolo-label-dir", type=Path, help="Prediction YOLO label directory.")
    parser.add_argument("--outdir", type=Path, required=True, help="Output directory for overlays and summaries.")
    parser.add_argument("--image-root", type=Path, default=None, help="Optional root containing spectrogram tile PNGs.")
    parser.add_argument("--max-images", type=int, default=24, help="Maximum overlay images to write. Use 0 for all.")
    parser.add_argument("--select", choices=("first", "random", "worst"), default="first")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--default-yolo-score", type=float, default=1.0)
    parser.add_argument(
        "--generate-random-predictions",
        action="store_true",
        help="Generate jittered random predictions from the ground-truth boxes and compare against them.",
    )
    parser.add_argument("--random-output", type=Path, default=None, help="Where to write generated random predictions.")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument("--random-drop-probability", type=float, default=0.15)
    parser.add_argument("--random-jitter-fraction", type=float, default=0.08)
    parser.add_argument("--random-size-jitter-fraction", type=float, default=0.08)
    parser.add_argument("--random-false-positive-rate", type=float, default=0.20)
    parser.add_argument("--random-score-min", type=float, default=0.35)
    parser.add_argument("--random-score-max", type=float, default=0.98)
    args = parser.parse_args()
    if not 0.0 <= args.iou_threshold <= 1.0:
        parser.error("--iou-threshold must be between 0 and 1")
    if args.max_images < 0:
        parser.error("--max-images must be >= 0")
    if args.generate_random_predictions and (args.pred_coco or args.pred_yolo_label_dir):
        parser.error("--generate-random-predictions cannot be combined with prediction input")
    return args


def main() -> int:
    args = parse_args()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    gt_images, gt_boxes, categories, gt_data = load_coco(args.coco.resolve(), source="gt")
    pred_boxes: list[BoxRecord] = []
    pred_data: dict[str, Any] | None = None

    if args.generate_random_predictions:
        pred_data = generate_random_predictions(
            gt_data,
            seed=args.random_seed,
            drop_probability=args.random_drop_probability,
            jitter_fraction=args.random_jitter_fraction,
            size_jitter_fraction=args.random_size_jitter_fraction,
            false_positive_rate=args.random_false_positive_rate,
            score_min=args.random_score_min,
            score_max=args.random_score_max,
        )
        random_path = args.random_output.resolve() if args.random_output else outdir / "random_predictions.json"
        random_path.parent.mkdir(parents=True, exist_ok=True)
        random_path.write_text(json.dumps(pred_data, indent=2), encoding="utf-8")
        _, pred_boxes, _, _ = load_coco(random_path, source="pred")
    elif args.pred_coco:
        pred_images, pred_boxes, pred_categories, pred_data = load_coco(args.pred_coco.resolve(), source="pred")
        missing = sorted(set(pred_images) - set(gt_images))
        if missing:
            raise ValueError(f"Prediction COCO contains image IDs absent from ground truth: {missing[:10]}")
        if pred_categories:
            categories = {**pred_categories, **categories}
    elif args.pred_yolo_label_dir:
        pred_boxes = load_yolo_predictions(
            args.pred_yolo_label_dir.resolve(),
            gt_images,
            sorted(categories),
            default_score=args.default_yolo_score,
        )

    matches = match_boxes(gt_boxes, pred_boxes, iou_threshold=args.iou_threshold) if pred_boxes else []
    summary = summarize_matches(matches, categories) if pred_boxes else {
        "overall": {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "mean_iou": 0.0},
        "categories": {},
    }

    gt_by_image = group_by_image(gt_boxes)
    pred_by_image = group_by_image(pred_boxes)
    images_to_render = select_images(
        gt_images,
        gt_by_image,
        pred_by_image,
        matches,
        max_images=args.max_images,
        mode=args.select,
        seed=args.random_seed,
    )
    render_overlays(
        outdir=outdir,
        images=images_to_render,
        gt_by_image=gt_by_image,
        pred_by_image=pred_by_image,
        matches=matches,
        categories=categories,
        image_root=args.image_root.resolve() if args.image_root else None,
        title_prefix="GT/PRED " if pred_boxes else "GT ",
    )

    summary_out = {
        "ground_truth": str(args.coco.resolve()),
        "prediction_source": (
            str(args.pred_coco.resolve())
            if args.pred_coco
            else str(args.pred_yolo_label_dir.resolve())
            if args.pred_yolo_label_dir
            else "random"
            if args.generate_random_predictions
            else None
        ),
        "iou_threshold": args.iou_threshold,
        "ground_truth_boxes": len(gt_boxes),
        "prediction_boxes": len(pred_boxes),
        "rendered_images": len(images_to_render),
        "metrics": summary,
    }
    (outdir / "comparison_summary.json").write_text(json.dumps(summary_out, indent=2), encoding="utf-8")
    write_matches_csv(outdir / "matches.csv", matches, categories)

    print(f"Ground truth boxes: {len(gt_boxes)}")
    print(f"Prediction boxes: {len(pred_boxes)}")
    print(f"Rendered overlays: {len(images_to_render)}")
    if pred_boxes:
        overall = summary["overall"]
        print(
            "Overall: "
            f"TP={overall['tp']} FP={overall['fp']} FN={overall['fn']} "
            f"precision={overall['precision']:.4f} recall={overall['recall']:.4f} "
            f"mean_iou={overall['mean_iou']:.4f}"
        )
    print(f"Overlays: {outdir / 'overlays'}")
    print(f"Summary: {outdir / 'comparison_summary.json'}")
    if args.generate_random_predictions:
        print(f"Random predictions: {args.random_output.resolve() if args.random_output else outdir / 'random_predictions.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
