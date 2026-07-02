"""
    create a fixed, stratified train/val/test split from the EuroSAT manifest

    - reads data/splits/manifest.csv (written by prepare_eurosat.py)
    - stratified split per class, 70/15/15
    - writes train.csv / val.csv / test.csv to data/splits

    all other scripts (training, robustness, interpretability, ...) should
    read these split files instead of re-splitting, so every model sees the
    exact same train/val/test images

    run:
        uv run python scripts/make_splits.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


SEED = 42
TRAIN_FRAC = 0.7
VAL_FRAC = 0.15
# remaining 0.15 goes to test


def make_splits(manifest_path: Path, splits_dir: Path) -> None:
    manifest = pd.read_csv(manifest_path)

    train_df, temp_df = train_test_split(
        manifest,
        train_size=TRAIN_FRAC,
        stratify=manifest["class_name"],
        random_state=SEED,
    )

    val_share_of_temp = VAL_FRAC / (1 - TRAIN_FRAC)

    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_share_of_temp,
        stratify=temp_df["class_name"],
        random_state=SEED,
    )

    train_df.to_csv(splits_dir / "train.csv", index=False)
    val_df.to_csv(splits_dir / "val.csv", index=False)
    test_df.to_csv(splits_dir / "test.csv", index=False)

    print(f"train: {len(train_df):5d} images")
    print(f"val:   {len(val_df):5d} images")
    print(f"test:  {len(test_df):5d} images")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a fixed train/val/test split from manifest.csv")
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path("data/splits"),
        help="Directory containing manifest.csv, also used as output directory",
    )

    args = parser.parse_args()
    splits_dir: Path = args.splits_dir

    manifest_path = splits_dir / "manifest.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"{manifest_path} not found. Run scripts/prepare_eurosat.py first."
        )

    make_splits(manifest_path, splits_dir)

    print(f"\nWrote split files to {splits_dir}")


if __name__ == "__main__":
    main()