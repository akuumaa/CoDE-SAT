"""
    evaluate the stage-1 checkpoints on the (perturbed) test set

    - clean + 10 perturbations: gaussian noise, rotation 15/30/45,
      blur k=3/5/7, occlusion 10/25/40%
    - writes aggregate metrics per model to outputs/eval/*.json
    - writes per-image predictions (path, label, pred) as csv,
      used for confusion matrices and the mcnemar test

    run (full, on the pod):
        uv run python scripts/evaluate_robustness.py

    smoke test (local cpu, few batches):
        uv run python scripts/evaluate_robustness.py \
            --models convnext_tiny.fb_in1k --only clean rotation15 --limit-batches 3
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import timm
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode
from tqdm import tqdm

torch.multiprocessing.set_sharing_strategy("file_system")

ROOT = Path(__file__).resolve().parent.parent

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# perturbation definitions also end up in the result json so the report
# can look the parameters up later
PERTURBATIONS = {
    "clean": {"kind": "none"},
    "noise": {"kind": "noise", "std": 0.1},
    "rotation15": {"kind": "rotation", "angle": 15},
    "rotation30": {"kind": "rotation", "angle": 30},
    "rotation45": {"kind": "rotation", "angle": 45},
    "blur3": {"kind": "blur", "kernel": 3},
    "blur5": {"kind": "blur", "kernel": 5},
    "blur7": {"kind": "blur", "kernel": 7},
    "occlusion10": {"kind": "occlusion", "fraction": 0.10},
    "occlusion25": {"kind": "occlusion", "fraction": 0.25},
    "occlusion40": {"kind": "occlusion", "fraction": 0.40},
}

CHECKPOINT_SUFFIX = "_sgd_stage1_best.pt"


class PerturbedEuroSAT(Dataset):
    """test split with one perturbation applied after resize, before normalization"""

    def __init__(self, frame: pd.DataFrame, perturbation: dict, seed: int = 42):
        self.frame = frame.reset_index(drop=True)
        self.perturbation = perturbation
        self.seed = seed

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]

        image = Image.open(ROOT / row["path"]).convert("RGB")
        image = TF.resize(image, [224, 224])

        kind = self.perturbation["kind"]

        if kind == "rotation":
            image = TF.rotate(
                image,
                self.perturbation["angle"],
                interpolation=InterpolationMode.BILINEAR,
            )
        elif kind == "blur":
            # sigma from kernel size, opencv convention
            k = self.perturbation["kernel"]
            sigma = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
            image = TF.gaussian_blur(image, kernel_size=k, sigma=[sigma])

        tensor = TF.to_tensor(image)

        # noise and occlusion are seeded per image, so all models see
        # exactly the same perturbation instance
        if kind == "noise":
            generator = torch.Generator().manual_seed(self.seed * 100003 + index)
            noise = torch.randn(tensor.shape, generator=generator)
            tensor = (tensor + noise * self.perturbation["std"]).clamp(0.0, 1.0)
        elif kind == "occlusion":
            generator = torch.Generator().manual_seed(self.seed * 100003 + index)
            side = round(math.sqrt(self.perturbation["fraction"]) * 224)
            top = torch.randint(0, 224 - side + 1, (1,), generator=generator).item()
            left = torch.randint(0, 224 - side + 1, (1,), generator=generator).item()

            # imagenet mean = zero signal after normalization
            fill = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
            tensor[:, top:top + side, left:left + side] = fill

        tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)

        label = int(row["label"])

        return tensor, label, index


@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device, limit_batches=None, desc="eval"):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []
    all_indices = []

    for batch_index, (images, labels, indices) in enumerate(tqdm(data_loader, desc=desc)):
        if limit_batches is not None and batch_index >= limit_batches:
            break

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        all_indices.extend(indices.tolist())

    num_images = len(all_labels)

    metrics = {
        "loss": total_loss / num_images,
        "acc": accuracy_score(all_labels, all_predictions),
        "f1": f1_score(all_labels, all_predictions, average="macro"),
        "num_images": num_images,
    }

    return metrics, all_indices, all_labels, all_predictions


def find_checkpoints(checkpoint_dir: Path, model_filter):
    checkpoints = {}

    for path in sorted(checkpoint_dir.glob(f"*{CHECKPOINT_SUFFIX}")):
        model_name = path.name.removesuffix(CHECKPOINT_SUFFIX)

        if model_filter and model_name not in model_filter:
            continue

        checkpoints[model_name] = path

    return checkpoints


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT / "results" / "stage1_checkpoints",
        help="directory with the stage-1 *_best.pt checkpoints",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "eval",
        help="where json results and prediction csvs are written",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="timm model names to evaluate (default: all found checkpoints)",
    )

    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        choices=list(PERTURBATIONS),
        help="subset of perturbations to run (default: all)",
    )

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="only evaluate this many batches per run (local smoke test)",
    )

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    test_frame = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    classes_df = pd.read_csv(ROOT / "data" / "splits" / "classes.txt").sort_values("label")
    classes = classes_df["class_name"].tolist()

    checkpoints = find_checkpoints(args.checkpoint_dir, args.models)

    if not checkpoints:
        raise SystemExit(f"no checkpoints matching *{CHECKPOINT_SUFFIX} in {args.checkpoint_dir}")

    perturbation_names = args.only or list(PERTURBATIONS)

    predictions_dir = args.output_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)

    loss_fn = nn.CrossEntropyLoss()

    for model_name, checkpoint_path in checkpoints.items():
        print(f"\n=== {model_name} ===")
        print(f"checkpoint: {checkpoint_path}")

        # pretrained=False, the weights come entirely from the checkpoint
        model = timm.create_model(model_name, pretrained=False, num_classes=len(classes))
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model = model.to(device)

        results = {}

        for pert_name in perturbation_names:
            perturbation = PERTURBATIONS[pert_name]

            dataset = PerturbedEuroSAT(test_frame, perturbation, seed=args.seed)

            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=device == "cuda",
            )

            metrics, indices, labels, predictions = evaluate(
                model,
                loader,
                loss_fn,
                device,
                limit_batches=args.limit_batches,
                desc=f"{model_name} / {pert_name}",
            )

            results[pert_name] = {**perturbation, **metrics}

            print(f"{pert_name:12s} acc {metrics['acc']:.4f}  f1 {metrics['f1']:.4f}")

            predictions_frame = pd.DataFrame({
                "path": test_frame.iloc[indices]["path"].tolist(),
                "label": labels,
                "pred": predictions,
            })

            csv_path = predictions_dir / f"{model_name}__{pert_name}.csv"
            predictions_frame.to_csv(csv_path, index=False)

        summary = {
            "model": model_name,
            "checkpoint": str(checkpoint_path),
            "classes": classes,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "limit_batches": args.limit_batches,
            "results": results,
        }

        json_path = args.output_dir / f"{model_name}_stage3_robustness.json"

        with open(json_path, "w") as file:
            json.dump(summary, file, indent=4)

        print(f"saved: {json_path}")

    print(f"\nper-image predictions: {predictions_dir}")


if __name__ == "__main__":
    main()