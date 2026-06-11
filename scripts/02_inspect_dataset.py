#!/usr/bin/env python3
"""Inspect the D-Fire dataset images in data/raw/{train,val,test}.

For each split: count images, check file formats, collect dimension
statistics, detect corrupted/unreadable images, and save a 4x4 sample
grid of images to reports/sample_grid.png.
"""

import random
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from _common import RAW_DIR, REPORTS_DIR, SPLITS, list_images

GRID_SIZE = 4
SAMPLE_COUNT = GRID_SIZE * GRID_SIZE


def inspect_split(split):
    img_dir = RAW_DIR / split / "images"
    images = list_images(split)

    if not img_dir.is_dir():
        return {"split": split, "present": False}

    formats = {}
    widths, heights = [], []
    corrupted = []
    readable = []

    for path in images:
        formats[path.suffix.lower()] = formats.get(path.suffix.lower(), 0) + 1

        img = cv2.imread(str(path))
        if img is None:
            corrupted.append(path.name)
            continue

        h, w = img.shape[:2]
        widths.append(w)
        heights.append(h)
        readable.append(path)

    return {
        "split": split,
        "present": True,
        "count": len(images),
        "formats": formats,
        "corrupted": corrupted,
        "widths": widths,
        "heights": heights,
        "readable": readable,
    }


def stats_str(values):
    if not values:
        return "n/a"
    return f"min={min(values)} max={max(values)} mean={sum(values) / len(values):.1f}"


def print_summary(results):
    print("\n" + "=" * 60)
    print("Dataset inspection summary")
    print("=" * 60)

    for r in results:
        print(f"\n--- {r['split']} ---")
        if not r["present"]:
            print("  (split directory not found, skipped)")
            continue

        fmt_str = ", ".join(f"{k}={v}" for k, v in sorted(r["formats"].items())) or "none"
        print(f"  Images        : {r['count']}")
        print(f"  Formats       : {fmt_str}")
        print(f"  Width  (px)   : {stats_str(r['widths'])}")
        print(f"  Height (px)   : {stats_str(r['heights'])}")

        if r["corrupted"]:
            print(f"  Corrupted     : {len(r['corrupted'])}")
            for name in r["corrupted"][:10]:
                print(f"    - {name}")
            if len(r["corrupted"]) > 10:
                print(f"    ... and {len(r['corrupted']) - 10} more")
        else:
            print("  Corrupted     : 0")

    print("\n" + "-" * 60)
    print(f"{'Split':<10}{'Images':>10}{'Corrupted':>12}{'Formats':>20}")
    print("-" * 60)
    for r in results:
        if not r["present"]:
            print(f"{r['split']:<10}{'-':>10}{'-':>12}{'n/a':>20}")
            continue
        fmt_str = ",".join(sorted(r["formats"]))
        print(f"{r['split']:<10}{r['count']:>10}{len(r['corrupted']):>12}{fmt_str:>20}")
    print("=" * 60)


def save_sample_grid(results):
    pool = []
    for r in results:
        if r["present"]:
            pool.extend(r["readable"])

    if not pool:
        print("\nNo readable images found, skipping sample grid.")
        return

    random.seed(42)
    sample = random.sample(pool, min(SAMPLE_COUNT, len(pool)))

    fig, axes = plt.subplots(GRID_SIZE, GRID_SIZE, figsize=(12, 12))
    for ax in axes.flat:
        ax.axis("off")

    for ax, path in zip(axes.flat, sample):
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(path.name, fontsize=8)
        ax.axis("off")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "sample_grid.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"\nSample grid saved to {out_path}")


def main():
    results = [inspect_split(split) for split in SPLITS]
    print_summary(results)
    save_sample_grid(results)

    any_present = any(r["present"] for r in results)
    if not any_present:
        print("\nFailure: no dataset splits found under data/raw/.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
