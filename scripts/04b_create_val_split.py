#!/usr/bin/env python3
"""Carve a validation split out of data/raw/train for the D-Fire dataset.

D-Fire ships with only train/ and test/, but YOLOv8 needs a val/ split
during training. This script pairs each train image with its label,
determines a per-image stratum (dominant class, or "background" /
"mixed"), and uses a stratified train_test_split (test_size=0.2,
random_state=42) to MOVE 20% of the train files into data/raw/val/.

After moving, it prints a verification table, and saves an updated
reports/split_summary.json and reports/split_summary.png.
"""

import json
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from _common import RAW_DIR, REPORTS_DIR, SPLITS, analyze_split_labels, list_images

VAL_FRACTION = 0.2
RANDOM_STATE = 42


def count_classes(label_path):
    """Return (fire_count, smoke_count) from a YOLO label file."""
    fire = smoke = 0
    if not label_path.is_file():
        return fire, smoke

    for line in label_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            class_id = int(parts[0])
        except (ValueError, IndexError):
            continue
        if class_id == 0:
            fire += 1
        elif class_id == 1:
            smoke += 1

    return fire, smoke


def dominant_stratum(fire, smoke):
    if fire == 0 and smoke == 0:
        return "background"
    if fire > smoke:
        return "fire"
    if smoke > fire:
        return "smoke"
    return "mixed"


def split_stats(split):
    result = analyze_split_labels(split)
    return {
        "images": result["num_images"],
        "fire_boxes": result["class_counts"][0],
        "smoke_boxes": result["class_counts"][1],
        "no_annotation_images": len(result["missing_labels"]) + len(result["empty_labels"]),
    }


def main():
    train_lbl_dir = RAW_DIR / "train" / "labels"
    val_img_dir = RAW_DIR / "val" / "images"
    val_lbl_dir = RAW_DIR / "val" / "labels"

    if val_img_dir.is_dir() and any(val_img_dir.iterdir()):
        print("data/raw/val/images already contains files; aborting to avoid re-splitting.")
        return 1

    images = list_images("train")
    if not images:
        print("No training images found under data/raw/train/images.")
        return 1

    pairs = []
    strata = []
    for img_path in images:
        lbl_path = train_lbl_dir / (img_path.stem + ".txt")
        fire, smoke = count_classes(lbl_path)
        pairs.append((img_path, lbl_path))
        strata.append(dominant_stratum(fire, smoke))

    _, val_pairs = train_test_split(
        pairs, test_size=VAL_FRACTION, random_state=RANDOM_STATE, stratify=strata
    )

    val_img_dir.mkdir(parents=True, exist_ok=True)
    val_lbl_dir.mkdir(parents=True, exist_ok=True)

    for img_path, lbl_path in val_pairs:
        shutil.move(str(img_path), str(val_img_dir / img_path.name))
        if lbl_path.is_file():
            shutil.move(str(lbl_path), str(val_lbl_dir / lbl_path.name))

    print(f"Moved {len(val_pairs)} image/label pairs from train/ to val/.")

    # Verification table
    stats = {split: split_stats(split) for split in SPLITS}

    print("\n" + "=" * 70)
    print("Train/Val/Test split verification")
    print("=" * 70)
    print(
        f"{'Split':<10}{'Images':>10}{'Fire boxes':>14}"
        f"{'Smoke boxes':>14}{'No-annot images':>18}"
    )
    for split in SPLITS:
        s = stats[split]
        print(
            f"{split:<10}{s['images']:>10}{s['fire_boxes']:>14}"
            f"{s['smoke_boxes']:>14}{s['no_annotation_images']:>18}"
        )
    print("=" * 70)

    # JSON report
    total_images = sum(stats[s]["images"] for s in SPLITS)
    report = {
        "image_counts": {s: stats[s]["images"] for s in SPLITS},
        "split_ratio_pct": {
            s: round(100 * stats[s]["images"] / total_images, 2) if total_images else 0.0
            for s in SPLITS
        },
        "class_distribution": {
            s: {"fire": stats[s]["fire_boxes"], "smoke": stats[s]["smoke_boxes"]} for s in SPLITS
        },
        "no_annotation_images": {s: stats[s]["no_annotation_images"] for s in SPLITS},
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "split_summary.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nJSON report saved to {json_path}")

    # Bar chart
    x = range(len(SPLITS))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([p - width for p in x], [stats[s]["fire_boxes"] for s in SPLITS], width, label="fire boxes")
    ax.bar([p for p in x], [stats[s]["smoke_boxes"] for s in SPLITS], width, label="smoke boxes")
    ax.bar(
        [p + width for p in x],
        [stats[s]["no_annotation_images"] for s in SPLITS],
        width,
        label="no-annotation images",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels(SPLITS)
    ax.set_ylabel("Count")
    ax.set_title("Class distribution per split (after val split)")
    ax.legend()
    fig.tight_layout()

    chart_path = REPORTS_DIR / "split_summary.png"
    fig.savefig(chart_path, dpi=100)
    plt.close(fig)
    print(f"Bar chart saved to {chart_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
