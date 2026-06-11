#!/usr/bin/env python3
"""Check train/val/test split integrity for the D-Fire dataset.

Counts images and labels per split, verifies no filename appears in more
than one split, prints the split ratio, and saves a class-distribution
bar chart to reports/split_summary.png plus a JSON report to
reports/split_summary.json.
"""

import json
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import CLASS_NAMES, RAW_DIR, REPORTS_DIR, SPLITS, analyze_split_labels, list_images


def count_labels(split):
    lbl_dir = RAW_DIR / split / "labels"
    if not lbl_dir.is_dir():
        return 0
    return sum(1 for p in lbl_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt")


def find_cross_split_duplicates(split_stems):
    """Return {filename_stem: [splits...]} for stems present in >1 split."""
    seen = {}
    for split, stems in split_stems.items():
        for stem in stems:
            seen.setdefault(stem, []).append(split)
    return {stem: splits for stem, splits in seen.items() if len(splits) > 1}


def main():
    label_results = {split: analyze_split_labels(split) for split in SPLITS}
    image_counts = {}
    label_counts = {}
    split_stems = {}

    for split in SPLITS:
        images = list_images(split)
        image_counts[split] = len(images)
        label_counts[split] = count_labels(split)
        split_stems[split] = {p.stem for p in images}

    print("\n" + "=" * 60)
    print("Split integrity summary")
    print("=" * 60)
    print(f"{'Split':<10}{'Images':>10}{'Labels':>10}")
    for split in SPLITS:
        print(f"{split:<10}{image_counts[split]:>10}{label_counts[split]:>10}")

    total_images = sum(image_counts.values())
    print("\nSplit ratio:")
    if total_images == 0:
        print("  No images found.")
    else:
        for split in SPLITS:
            pct = 100 * image_counts[split] / total_images
            print(f"  {split:<6}: {pct:5.1f}% ({image_counts[split]} images)")

    duplicates = find_cross_split_duplicates(split_stems)
    print("\nCross-split duplicate filenames:")
    if duplicates:
        for stem, splits in list(duplicates.items())[:10]:
            print(f"  - {stem}: appears in {', '.join(splits)}")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more")
    else:
        print("  None found.")
    print("=" * 60)

    # Bar chart of class distribution per split
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    present_splits = [s for s in SPLITS if image_counts[s] > 0]
    if present_splits:
        x = range(len(present_splits))
        width = 0.35
        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (class_id, class_name) in enumerate(CLASS_NAMES.items()):
            counts = [label_results[s]["class_counts"][class_id] for s in present_splits]
            offset = (i - 0.5) * width
            ax.bar([p + offset for p in x], counts, width, label=class_name)

        ax.set_xticks(list(x))
        ax.set_xticklabels(present_splits)
        ax.set_ylabel("Bounding box count")
        ax.set_title("Class distribution per split")
        ax.legend()
        fig.tight_layout()

        chart_path = REPORTS_DIR / "split_summary.png"
        fig.savefig(chart_path, dpi=100)
        plt.close(fig)
        print(f"\nBar chart saved to {chart_path}")
    else:
        print("\nNo data found, skipping bar chart.")

    # JSON report
    report = {
        "image_counts": image_counts,
        "label_counts": label_counts,
        "split_ratio_pct": {
            split: round(100 * image_counts[split] / total_images, 2) if total_images else 0.0
            for split in SPLITS
        },
        "cross_split_duplicates": {stem: splits for stem, splits in duplicates.items()},
        "class_distribution": {
            split: {CLASS_NAMES[c]: label_results[split]["class_counts"][c] for c in CLASS_NAMES}
            for split in SPLITS
        },
    }
    json_path = REPORTS_DIR / "split_summary.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"JSON report saved to {json_path}")

    if total_images == 0:
        print("\nFailure: no dataset splits found under data/raw/.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
