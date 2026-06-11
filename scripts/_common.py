"""Shared constants and helpers for the dataset inspection/validation scripts."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
REPORTS_DIR = ROOT / "reports"

SPLITS = ("train", "val", "test")
CLASS_NAMES = {0: "fire", 1: "smoke"}
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def list_images(split):
    """Return sorted image file paths for a split, or [] if the split is absent."""
    img_dir = RAW_DIR / split / "images"
    if not img_dir.is_dir():
        return []
    return sorted(
        p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def analyze_split_labels(split):
    """Validate YOLO labels for every image in a split.

    Returns counts/lists of missing labels, empty labels, malformed lines
    (wrong field count, bad class id, out-of-range coordinates), and the
    per-class bounding-box distribution.
    """
    lbl_dir = RAW_DIR / split / "labels"
    images = list_images(split)

    missing_labels = []
    empty_labels = []
    invalid_count_lines = []
    invalid_class_lines = []
    out_of_range_lines = []
    class_counts = {0: 0, 1: 0}

    for img_path in images:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.is_file():
            missing_labels.append(img_path.name)
            continue

        lines = [l.strip() for l in lbl_path.read_text().splitlines() if l.strip()]
        if not lines:
            empty_labels.append(img_path.name)
            continue

        for line_no, line in enumerate(lines, start=1):
            parts = line.split()
            if len(parts) != 5:
                invalid_count_lines.append((img_path.name, line_no, line))
                continue

            try:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
            except ValueError:
                invalid_class_lines.append((img_path.name, line_no, line))
                continue

            if class_id not in CLASS_NAMES:
                invalid_class_lines.append((img_path.name, line_no, line))
                continue

            if not all(0.0 <= c <= 1.0 for c in coords):
                out_of_range_lines.append((img_path.name, line_no, line))
                continue

            class_counts[class_id] += 1

    return {
        "split": split,
        "num_images": len(images),
        "missing_labels": missing_labels,
        "empty_labels": empty_labels,
        "invalid_count_lines": invalid_count_lines,
        "invalid_class_lines": invalid_class_lines,
        "out_of_range_lines": out_of_range_lines,
        "class_counts": class_counts,
    }
