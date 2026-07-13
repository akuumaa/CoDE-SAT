"""
    pairwise statistical comparison of the three models (mcnemar test)

    - reads the per-image clean predictions written by evaluate_robustness.py
      (results/stage3/predictions/*__clean.csv)
    - for each model pair, counts the images where exactly one of the two
      models is correct and runs an exact binomial test on those counts
    - answers: is the accuracy difference real or just chance?

    run:
        uv run python scripts/compare_models.py
"""

from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
PREDICTIONS = ROOT / "results" / "stage3" / "predictions"

MODELS = [
    "convnext_tiny.fb_in1k",
    "deit_small_patch16_224.fb_in1k",
    "deit_small_distilled_patch16_224.fb_in1k",
]


def load_correct(model: str) -> pd.Series:
    frame = pd.read_csv(PREDICTIONS / f"{model}__clean.csv")
    return (frame["label"] == frame["pred"]).rename(model)


def main():
    correct = {m: load_correct(m) for m in MODELS}

    print(f"{'Modell A':42s} {'Modell B':42s} {'b':>4s} {'c':>4s} {'p-Wert':>8s}")

    for a, b in combinations(MODELS, 2):
        # b_count: A richtig, B falsch; c_count: A falsch, B richtig
        b_count = int((correct[a] & ~correct[b]).sum())
        c_count = int((~correct[a] & correct[b]).sum())

        n = b_count + c_count
        p = binomtest(b_count, n, 0.5).pvalue if n > 0 else float("nan")

        print(f"{a:42s} {b:42s} {b_count:4d} {c_count:4d} {p:8.4f}")


if __name__ == "__main__":
    main()