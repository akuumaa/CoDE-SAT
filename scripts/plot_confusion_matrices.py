"""
    confusion matrices of the three models on the clean testset

    - reads the per-image clean predictions written by evaluate_robustness.py
      (results/stage3/predictions/*__clean.csv)
    - color = row-normalized fraction of the true class, numbers = absolute
      image counts; zero cells stay empty so the errors stand out
    - writes results/figures/stage3_confusion_matrices.{png,pdf}

    run:
        uv run python scripts/plot_confusion_matrices.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "results" / "stage3" / "predictions"

MODELS = {
    "convnext_tiny.fb_in1k": "ConvNeXt-Tiny",
    "deit_small_patch16_224.fb_in1k": "DeiT-Small",
    "deit_small_distilled_patch16_224.fb_in1k": "DeiT-Small (distilled)",
}

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})


def main():
    classes = (
        pd.read_csv(ROOT / "data" / "splits" / "classes.txt")
        .sort_values("label")["class_name"].tolist()
    )
    n = len(classes)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6.2))

    for ax, (model, label) in zip(axes, MODELS.items()):
        frame = pd.read_csv(PREDICTIONS / f"{model}__clean.csv")

        matrix = np.zeros((n, n), dtype=int)
        np.add.at(matrix, (frame["label"], frame["pred"]), 1)
        row_norm = matrix / matrix.sum(axis=1, keepdims=True).clip(min=1)

        ax.imshow(row_norm, cmap="Blues", vmin=0, vmax=1)
        ax.grid(False)
        for i in range(n):
            for j in range(n):
                if matrix[i, j]:
                    ax.text(j, i, matrix[i, j], ha="center", va="center", fontsize=7,
                            color="white" if row_norm[i, j] > 0.5 else "#333333")

        accuracy = 100 * (frame["label"] == frame["pred"]).mean()
        ax.set_title(f"{label} — Acc {accuracy:.2f} %", loc="left", fontsize=11)
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(classes if ax is axes[0] else [""] * n, fontsize=8)

    axes[0].set_ylabel("Wahre Klasse", fontsize=9)
    fig.supxlabel("Vorhersage", fontsize=9)

    fig.suptitle("Confusion-Matrices (Stufe 1, bestes Checkpoint je Modell, Testset ohne Störung)",
                 x=0.01, y=0.99, ha="left", fontsize=14)
    fig.text(0.01, 0.935, "Zahlen = Bildanzahl, Farbe = Anteil an der wahren Klasse (zeilennormiert)",
             color="#666666", fontsize=10)
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))

    figures_dir = ROOT / "results" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for ext in ["png", "pdf"]:
        fig.savefig(figures_dir / f"stage3_confusion_matrices.{ext}", dpi=200)

    print(f"saved to {figures_dir}/stage3_confusion_matrices.png")


if __name__ == "__main__":
    main()
