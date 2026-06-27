"""
    script to prepare the EuroSat dataset

    - download
    - extract to /data/raw/EuroSAAT_RGB
    - check classes
    - count pics per classe
    - writes manifest.csv
    - write classes.txt 
    - write data_summary.md

    run:
        uv run python scripts/prepare_eurosat.py
    run with override:
        uv run python scripts/prepare_eurosat.py --force
"""

from __future__ import annotations

import argparse
import csv
import shutil
import urllib.request
import zipfile
from pathlib import Path


EUROSAT_RGB_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"

EXPECTED_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]


def download_dataset(zip_path: Path) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if zip_path.exists():
        print(f"Zip already exists: {zip_path}")
        return

    print("Downloading EuroSAT RGB...")
    print(f"Source: {EUROSAT_RGB_URL}")
    print(f"Target: {zip_path}")

    urllib.request.urlretrieve(EUROSAT_RGB_URL, zip_path)

    print("Download finished.")


def extract_dataset(zip_path: Path, raw_dir: Path, force: bool = False) -> Path:
    dataset_root = raw_dir / "EuroSAT_RGB"

    if dataset_root.exists() and not force:
        print(f"Dataset already extracted: {dataset_root}")
        return dataset_root

    if dataset_root.exists() and force:
        print(f"Removing existing dataset: {dataset_root}")
        shutil.rmtree(dataset_root)

    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {zip_path} to {raw_dir}...")

    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(raw_dir)

    print("Extraction finished.")

    return dataset_root


def count_images_per_class(dataset_root: Path) -> dict[str, int]:
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {dataset_root}\n"
            "Expected the zip to extract into data/raw/EuroSAT_RGB."
        )

    class_counts: dict[str, int] = {}

    print("\nChecking classes and counting images:")

    for class_name in sorted(EXPECTED_CLASSES):
        class_dir = dataset_root / class_name

        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class folder: {class_dir}")

        images = sorted(class_dir.glob("*.jpg"))
        class_counts[class_name] = len(images)

        print(f"  {class_name:22s} {len(images):5d} images")

    total_images = sum(class_counts.values())
    print(f"\nTotal images: {total_images}")
    print("Dataset class check passed.")

    return class_counts


def write_classes_file(classes_path: Path, class_counts: dict[str, int]) -> None:
    classes_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_classes = sorted(EXPECTED_CLASSES)

    with classes_path.open("w", encoding="utf-8") as file:
        file.write("label,class_name,num_images\n")

        for label, class_name in enumerate(sorted_classes):
            num_images = class_counts[class_name]
            file.write(f"{label},{class_name},{num_images}\n")

    print(f"Wrote classes file: {classes_path}")


def write_manifest_file(dataset_root: Path, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    sorted_classes = sorted(EXPECTED_CLASSES)
    class_to_label = {class_name: label for label, class_name in enumerate(sorted_classes)}

    rows = []

    for class_name in sorted_classes:
        class_dir = dataset_root / class_name
        images = sorted(class_dir.glob("*.jpg"))

        for image_path in images:
            rows.append(
                {
                    "path": image_path.as_posix(),
                    "label": class_to_label[class_name],
                    "class_name": class_name,
                }
            )

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label", "class_name"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote manifest file: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare EuroSAT RGB.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base data directory. Default: data",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract the dataset if it already exists.",
    )

    args = parser.parse_args()
    data_dir: Path = args.data_dir

    zip_path = data_dir / "downloads" / "EuroSAT_RGB.zip"
    raw_dir = data_dir / "raw"
    splits_dir = data_dir / "splits"

    dataset_root = raw_dir / "EuroSAT_RGB"
    classes_path = splits_dir / "classes.txt"
    manifest_path = splits_dir / "manifest.csv"

    download_dataset(zip_path)
    dataset_root = extract_dataset(zip_path, raw_dir, force=args.force)

    class_counts = count_images_per_class(dataset_root)

    write_classes_file(classes_path, class_counts)
    write_manifest_file(dataset_root, manifest_path)

    print("\nEuroSAT RGB is ready.")


if __name__ == "__main__":
    main()