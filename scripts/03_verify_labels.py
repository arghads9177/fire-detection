#!/usr/bin/env python3
"""Validate YOLO labels for the D-Fire dataset in data/raw/{train,val,test}.

For every image, check that a corresponding .txt label exists and that
each label line has 5 fields with class_id in {0, 1} (0=fire, 1=smoke)
and coordinates (cx, cy, w, h) within [0, 1]. Reports missing/empty
labels, malformed lines, and the per-split class distribution.
"""

import sys

from _common import CLASS_NAMES, SPLITS, analyze_split_labels


def print_issues(title, items, limit=10):
    print(f"  {title}: {len(items)}")
    for item in items[:limit]:
        if isinstance(item, tuple):
            name, line_no, line = item
            print(f"    - {name}:{line_no}: '{line}'")
        else:
            print(f"    - {item}")
    if len(items) > limit:
        print(f"    ... and {len(items) - limit} more")


def main():
    results = [analyze_split_labels(split) for split in SPLITS]

    print("\n" + "=" * 60)
    print("Label validation summary")
    print("=" * 60)

    any_present = False
    for r in results:
        print(f"\n--- {r['split']} ---")
        if r["num_images"] == 0:
            print("  (no images found, skipped)")
            continue
        any_present = True

        print(f"  Images                 : {r['num_images']}")
        print_issues("Missing labels", r["missing_labels"])
        print_issues("Empty labels", r["empty_labels"])
        print_issues("Invalid field count", r["invalid_count_lines"])
        print_issues("Invalid class id / value", r["invalid_class_lines"])
        print_issues("Out-of-range coordinates", r["out_of_range_lines"])

    print("\n" + "-" * 60)
    print("Class distribution (bounding box counts)")
    print("-" * 60)
    header = f"{'Split':<10}" + "".join(f"{name:>12}" for name in CLASS_NAMES.values()) + f"{'total':>12}"
    print(header)
    for r in results:
        if r["num_images"] == 0:
            continue
        counts = r["class_counts"]
        total = sum(counts.values())
        row = f"{r['split']:<10}" + "".join(f"{counts[c]:>12}" for c in CLASS_NAMES) + f"{total:>12}"
        print(row)
    print("=" * 60)

    if not any_present:
        print("\nFailure: no dataset splits found under data/raw/.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
