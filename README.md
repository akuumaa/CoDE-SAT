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

Best checkpoint: `outputs/checkpoints/`, metrics: `outputs/metrics/`.