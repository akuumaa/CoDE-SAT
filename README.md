# CoDE-SAT

Comparison of ConvNeXt and DeiT on EuroSAT.

## Setup

```bash
uv sync
```

## Data

```bash
uv run python scripts/prepare_eurosat.py
uv run python scripts/make_splits.py
```

## Training

```bash
uv run python train.py --model convnext_tiny.fb_in1k
```

## Flags

| Flag | Default | Description |
| --- | --- | --- |
| `--model` | `deit_tiny_patch16_224` | timm model name |
| `--epochs` | `20` | number of training epochs |
| `--batch-size` | `32` | batch size |
| `--lr` | `1e-4` | learning rate |
| `--num-workers` | `4` | number of dataloader worker processes |
| `--optimizer` | `adamw` | `adamw`, `sgd`, or `rmsprop` |
| `--momentum` | `0.9` | momentum (only used by `sgd`) |
| `--seed` | `42` | random seed |
| `--scheduler` | `cosine` | `cosine` or `none` |
| `--train-fraction` | `1.0` | stratified fraction of `train.csv` to use (val/test unchanged) |
| `--tag` | `""` | optional suffix for checkpoint/results filenames |



Best checkpoint: `outputs/checkpoints/`, metrics: `outputs/metrics/`.

## Results

Committed run results (metrics JSON, no checkpoints) live under `results/`,
grouped by model:

- `results/conv/` — ConvNeXt-Tiny
- `results/deit/` — DeiT-Small
- `results/deit_distilled/` — DeiT-Small distilled
