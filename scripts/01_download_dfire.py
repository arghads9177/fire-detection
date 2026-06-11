#!/usr/bin/env python3
"""Download the D-Fire dataset into data/raw/, preserving the
train/val/test/{images,labels} folder structure."""

import shutil
import sys
from pathlib import Path

import git
from tqdm import tqdm

REPO_URL = "https://github.com/gaia-solutions-on-demand/DFireDataset.git"
SPLITS = ("train", "val", "test")
SUBDIRS = ("images", "labels")

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

MANUAL_DOWNLOAD_MSG = """
The D-Fire dataset images/labels are not stored in the GitHub repo itself
(https://github.com/gaia-solutions-on-demand/DFireDataset) -- only docs and
utility code were retrieved. The actual pre-split train/val/test archives
are hosted externally. Download them manually and extract into data/raw/
so it looks like:

  data/raw/train/images/  data/raw/train/labels/
  data/raw/val/images/    data/raw/val/labels/
  data/raw/test/images/   data/raw/test/labels/

Sources (see the repo README for up-to-date links):
  - OneDrive (train/val/test sets):
    https://1drv.ms/f/c/c0bd25b6b048b01d/Ema8FFze8mFIlM1Hn81BUUgBE3vnnmK4SQxybS-nHRt2pA
  - Kaggle mirror (ready to use):
    https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo
"""


class CloneProgress(git.RemoteProgress):
    def __init__(self):
        super().__init__()
        self.pbar = tqdm(unit=" objects", desc="Cloning")

    def update(self, op_code, cur_count, max_count=None, message=""):
        if max_count:
            self.pbar.total = int(max_count)
        self.pbar.n = int(cur_count)
        self.pbar.refresh()


def has_existing_data() -> bool:
    return RAW_DIR.exists() and any(RAW_DIR.iterdir())


def clone_dataset_repo(tmp_dir: Path) -> bool:
    print(f"Cloning {REPO_URL} ...")
    try:
        git.Repo.clone_from(REPO_URL, tmp_dir, depth=1, progress=CloneProgress())
    except git.exc.GitCommandError as exc:
        print(f"  Failed to clone repository: {exc}")
        return False
    except Exception as exc:  # network errors, etc.
        print(f"  Network error while cloning: {exc}")
        return False
    return True


def extract_splits(tmp_dir: Path) -> bool:
    found_any = False
    for split in SPLITS:
        src = tmp_dir / split
        if not src.is_dir():
            continue
        for sub in SUBDIRS:
            sub_src = src / sub
            if sub_src.is_dir():
                dest = RAW_DIR / split / sub
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(sub_src, dest, dirs_exist_ok=True)
                found_any = True
    return found_any


def count_files(path: Path) -> int:
    if not path.is_dir():
        return 0
    return sum(1 for f in path.iterdir() if f.is_file())


def print_summary() -> bool:
    rows = []
    total_images = 0
    total_labels = 0
    for split in SPLITS:
        img_count = count_files(RAW_DIR / split / "images")
        lbl_count = count_files(RAW_DIR / split / "labels")
        rows.append((split, img_count, lbl_count))
        total_images += img_count
        total_labels += lbl_count

    print("\n" + "=" * 40)
    print(f"{'Split':<10}{'Images':>12}{'Labels':>12}")
    print("-" * 40)
    for split, img_count, lbl_count in rows:
        print(f"{split:<10}{img_count:>12}{lbl_count:>12}")
    print("-" * 40)
    print(f"{'Total':<10}{total_images:>12}{total_labels:>12}")
    print("=" * 40)

    return total_images > 0


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if has_existing_data():
        print(f"'{RAW_DIR}' already exists and contains data -- skipping download.")
        if print_summary():
            print("\nSuccess: D-Fire dataset is present in data/raw/.")
            return 0
        print("\nFailure: data/raw/ exists but no images were found in it.")
        return 1

    tmp_dir = RAW_DIR / "_tmp_clone"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    cloned = clone_dataset_repo(tmp_dir)
    found = extract_splits(tmp_dir) if cloned else False

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    print_summary()

    if not found:
        print(MANUAL_DOWNLOAD_MSG)
        print("Failure: could not automatically download the D-Fire dataset.")
        return 1

    print("\nSuccess: D-Fire dataset downloaded to data/raw/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
