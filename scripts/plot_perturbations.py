"""
    render one test image with every stage-3 perturbation applied

    - uses the exact same perturbation code and seeding as
      evaluate_robustness.py, so the figure shows the real
      perturbation instances from the evaluation
    - writes results/figures/stage3_perturbation_examples.{png,pdf}

    run:
        uv run python scripts/plot_perturbations.py
    pick a different image:
        uv run python scripts/plot_perturbations.py --index 123
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from evaluate_robustness import PERTURBATIONS, PerturbedEuroSAT, IMAGENET_MEAN, IMAGENET_STD

ROOT = Path(__file__).resolve().parent.parent

PANEL_TITLES = {
    "clean": "clean",
    "noise": "Noise σ=0.1",
    "rotation15": "Rotation 15°",
    "rotation30": "Rotation 30°",
    "rotation45": "Rotation 45°",
    "blur3": "Blur k=3",
    "blur5": "Blur k=5",
    "blur7": "Blur k=7",
    "occlusion10": "Occlusion 10 %",
    "occlusion25": "Occlusion 25 %",
    "occlusion40": "Occlusion 40 %",
}


def denormalize(tensor):
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int, default=0, help="row in test.csv to use")
    parser.add_argument("--seed", type=int, default=42, help="must match the evaluation run")
    args = parser.parse_args()

    test_frame = pd.read_csv(ROOT / "data" / "splits" / "test.csv")
    row = test_frame.iloc[args.index]

    fig, axes = plt.subplots(2, 6, figsize=(15, 6.0), gridspec_kw={"hspace": 0.35})
    axes = axes.flatten()

    for ax, (name, perturbation) in zip(axes, PERTURBATIONS.items()):
        dataset = PerturbedEuroSAT(test_frame, perturbation, seed=args.seed)
        tensor, _, _ = dataset[args.index]

        ax.imshow(denormalize(tensor))
        ax.set_title(PANEL_TITLES[name], fontsize=10, loc="left")
        ax.axis("off")

    # 11 panels on a 2x6 grid, hide the last slot
    axes[-1].axis("off")

    fig.suptitle(
        f"Störungen der Stufe 3 am Beispielbild ({row['class_name']}, {Path(row['path']).name})",
        x=0.01, y=0.99, ha="left", fontsize=14,
    )
    fig.text(0.01, 0.92, "identische Störungsinstanzen wie in der Evaluation "
             "(gleicher Seed, angewendet nach Resize auf 224×224)",
             color="#666666", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    figures_dir = ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for ext in ["png", "pdf"]:
        fig.savefig(figures_dir / f"stage3_perturbation_examples.{ext}", dpi=200)

    print(f"saved to {figures_dir}/stage3_perturbation_examples.png")


if __name__ == "__main__":
    main()