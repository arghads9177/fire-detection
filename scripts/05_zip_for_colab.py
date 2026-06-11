#!/usr/bin/env python3
"""Bundle the prepared D-Fire dataset and YOLOv8 config into a single zip
for upload to Colab.

Packs data/raw/{train,val,test}/{images,labels} and configs/dfire.yaml into
data/dfire_yolov8_ready.zip, preserving paths relative to the project root
so the dataset config's relative `path: ../data/raw` keeps working after
extraction.
"""

import sys
import zipfile

from tqdm import tqdm

from _common import RAW_DIR, ROOT

CONFIG_PATH = ROOT / "configs" / "dfire.yaml"
OUTPUT_ZIP = ROOT / "data" / "dfire_yolov8_ready.zip"


def collect_files():
    files = sorted(p for p in RAW_DIR.rglob("*") if p.is_file())
    files.append(CONFIG_PATH)
    return files


def main():
    if not RAW_DIR.is_dir():
        print(f"{RAW_DIR} not found.")
        return 1
    if not CONFIG_PATH.is_file():
        print(f"{CONFIG_PATH} not found.")
        return 1

    files = collect_files()

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in tqdm(files, desc="Zipping", unit="file"):
            zf.write(path, arcname=path.relative_to(ROOT))

    # Verify
    with zipfile.ZipFile(OUTPUT_ZIP) as zf:
        names = zf.namelist()
        bad = zf.testzip()

    if bad is not None:
        print(f"Corrupt entry found in zip: {bad}")
        return 1

    print(f"\nVerified zip contains {len(names)} files.")

    size_mb = OUTPUT_ZIP.stat().st_size / (1024 * 1024)
    print(f"Output: {OUTPUT_ZIP}")
    print(f"Zip size: {size_mb:.2f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
