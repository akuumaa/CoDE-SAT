# CoDE-SAT

Comparison of ConvNeXt and DeiT on EuroSAT.

## Setup

```bash
uv sync
```

## Projektablauf auf einen Blick

Die Experimente bauen aufeinander auf und sollten in dieser Reihenfolge laufen:

| # | Schritt | Skript | Ergebnisse landen in |
| --- | --- | --- | --- |
| 1 | Datensatz vorbereiten | `scripts/prepare_eurosat.py`, `scripts/make_splits.py` | `data/splits/` (lokal, nicht in git) |
| 2 | Stufe 0 - Optimizer-Vorstudie | `train.py` | `results/conv/`, `results/deit/` |
| 3 | Stufe 1 - Hauptvergleich (3 Modelle * 3 Seeds) | `train.py` | `results/stage1/`, Checkpoints in `results/stage1_checkpoints/` |
| 4 | Stufe 2 - Data Efficiency (10/25/50 % Daten) | `train.py` | `results/stage2/` |
| 5 | Stufe 3 - Robustheit | `scripts/evaluate_robustness.py`, `scripts/compare_models.py`, `scripts/plot_perturbations.py` | `results/stage3/` |
| 6 | Interpretierbarkeit | `scripts/interpretability.py` | `results/figures/interpretability/` |

## 1. Datensatz vorbereiten

### 1.1 EuroSAT herunterladen und prüfen - `scripts/prepare_eurosat.py`

- lädt `EuroSAT_RGB.zip` herunter (Zenodo) und entpackt nach `data/raw/EuroSAT_RGB`
- prüft, dass alle 10 Klassenordner vorhanden sind, und zählt die Bilder pro Klasse
- schreibt `data/splits/classes.txt` (Label ↔ Klassenname ↔ Bildanzahl) und `data/splits/manifest.csv` (Pfad, Label, Klassenname für jedes einzelne Bild)

| Flag | Default | Bedeutung |
| --- | --- | --- |
| `--data-dir` | `data` | Basis-Datenverzeichnis |
| `--force` | aus | entpackt neu, auch wenn `data/raw/EuroSAT_RGB` schon existiert |

```bash
uv run python scripts/prepare_eurosat.py
```

### 1.2 Train/Val/Test-Split erzeugen - `scripts/make_splits.py`

- liest `data/splits/manifest.csv`
- erzeugt einen festen, stratifizierten 70/15/15-Split (gleicher Klassenanteil in allen drei Teilmengen), Seed 42
- schreibt `train.csv`, `val.csv`, `test.csv` nach `data/splits/`

| Flag | Default | Bedeutung |
| --- | --- | --- |
| `--splits-dir` | `data/splits` | Verzeichnis mit `manifest.csv`, gleichzeitig Ausgabeverzeichnis |

```bash
uv run python scripts/make_splits.py
```

Am Ende muss dastehen: `train 18900`, `val 4049`, `test 4051`.

Alle weiteren Skripte lesen ausschließlich diese drei CSVs. `data/` ist nicht in git (siehe `.gitignore`), also muss jede/r Schritt 1.1 und 1.2 einmal lokal ausführen.

## 2. Training - `train.py`

`train.py` ist die gemeinsame Trainingsschleife für alle stages 0-2, welches Experiment es ist, steuern nur die Flags.

| Flag | Default | Bedeutung |
| --- | --- | --- |
| `--model` | `deit_tiny_patch16_224` | timm-Modellname |
| `--epochs` | `20` | Anzahl Trainingsepochen |
| `--batch-size` | `32` | Batch-Größe |
| `--lr` | `1e-4` | Lernrate |
| `--num-workers` | `4` | Anzahl Dataloader-Worker-Prozesse |
| `--optimizer` | `adamw` | `adamw`, `sgd`, oder `rmsprop` |
| `--momentum` | `0.9` | Momentum (nur bei `sgd`) |
| `--seed` | `42` | Zufalls-Seed |
| `--scheduler` | `cosine` | `cosine` oder `none` |
| `--train-fraction` | `1.0` | stratifizierter Anteil von `train.csv` (Val/Test bleiben unverändert) |
| `--tag` | `""` | optionales Suffix für Checkpoint-/Ergebnis-Dateinamen |

Checkpoint- und Ergebnis-Dateinamen ergeben sich automatisch aus der Run-Konfiguration:
`{model}_{optimizer}[_frac{train_fraction*100}][_{tag}]`, z.B. `convnext_tiny.fb_in1k_sgd_frac10_results.json` (10 % Daten) oder `convnext_tiny.fb_in1k_sgd_stage1_best.pt` (`--tag stage1`). So überschreiben sich unterschiedliche Optimizer, Datenanteile oder Tags nicht gegenseitig.

Checkpoints landen in `outputs/checkpoints/`, Metriken in `outputs/metrics/`. Beides ist nicht in git (`outputs/*` ist gitignored). Fertige Läufe werden manuell nach `results/<stufe>/` kopiert (siehe unten).

## 3. Stufe 0 - Optimizer-Vorstudie

Ziel: Optimizer für alle weiteren Stufen festlegen, bevor der Hauptvergleich läuft. 10 Epochen, ohne Scheduler, auf ConvNeXt-Tiny und DeiT-Small, jeweils AdamW/SGD/RMSprop:

```bash
uv run python train.py --model convnext_tiny.fb_in1k          --optimizer adamw   --lr 1e-4 --epochs 10 --scheduler none --num-workers 0
uv run python train.py --model convnext_tiny.fb_in1k          --optimizer sgd     --lr 1e-2 --epochs 10 --scheduler none --num-workers 0
uv run python train.py --model convnext_tiny.fb_in1k          --optimizer rmsprop --lr 1e-4 --epochs 10 --scheduler none --num-workers 0

uv run python train.py --model deit_small_patch16_224.fb_in1k --optimizer adamw   --lr 1e-4 --epochs 10 --scheduler none --num-workers 0
uv run python train.py --model deit_small_patch16_224.fb_in1k --optimizer sgd     --lr 1e-2 --epochs 10 --scheduler none --num-workers 0
uv run python train.py --model deit_small_patch16_224.fb_in1k --optimizer rmsprop --lr 1e-4 --epochs 10 --scheduler none --num-workers 0
```

`--num-workers 0` vermeidet einen Speicherfehler auf dem Trainings-Pod (siehe Kommentar in `train.py`). Lokal reicht auch der Default (`4`).

Die 6 resultierenden `*_results.json` von Hand aus `outputs/metrics/` nach `results/conv/` (ConvNeXt) bzw. `results/deit/` (DeiT-Small) kopieren.

## 4. Stufe 1 - Hauptvergleich

Alle 3 Modelle, 100 % Daten, SGD, Default-Epochen (20) mit Cosine-Scheduler. `--seed` ist standardmäßig `42`, muss für den ersten Durchlauf also gar nicht angegeben werden:

```bash
uv run python train.py --model convnext_tiny.fb_in1k                    --optimizer sgd --lr 1e-2 --tag stage1 --num-workers 0
uv run python train.py --model deit_small_patch16_224.fb_in1k           --optimizer sgd --lr 1e-2 --tag stage1 --num-workers 0
uv run python train.py --model deit_small_distilled_patch16_224.fb_in1k --optimizer sgd --lr 1e-2 --tag stage1 --num-workers 0
```

Für Mittelwert und Standardabweichung statt eines Einzellaufs werden dieselben 3 Befehle zusätzlich mit den Seeds 0 und 1 wiederholt, nur `--seed` und `--tag` ändern sich:

```bash
uv run python train.py --model convnext_tiny.fb_in1k --optimizer sgd --lr 1e-2 --tag stage1_seed0 --seed 0 --num-workers 0
uv run python train.py --model convnext_tiny.fb_in1k --optimizer sgd --lr 1e-2 --tag stage1_seed1 --seed 1 --num-workers 0
# gleiches für deit_small_patch16_224.fb_in1k und deit_small_distilled_patch16_224.fb_in1k
```

> 9 Läufe (3 Modelle * 3 Seeds)
> Ergebnis-JSONs -> `results/stage1/`
> Checkpoints (Seed 42, `*_stage1_best.pt`) -> `results/stage1_checkpoints/`, werden für Stufe 3 und die Interpretierbarkeit gebraucht

## 5. Stufe 2 - Data Efficiency

Alle 3 Modelle mit 50 %/25 %/10 % der Trainingsdaten, sonst identische Einstellungen wie Stufe 1:

```bash
uv run python train.py --model convnext_tiny.fb_in1k                    --optimizer sgd --lr 1e-2 --train-fraction 0.5  --num-workers 0
uv run python train.py --model convnext_tiny.fb_in1k                    --optimizer sgd --lr 1e-2 --train-fraction 0.25 --num-workers 0
uv run python train.py --model convnext_tiny.fb_in1k                    --optimizer sgd --lr 1e-2 --train-fraction 0.1  --num-workers 0

uv run python train.py --model deit_small_patch16_224.fb_in1k           --optimizer sgd --lr 1e-2 --train-fraction 0.5  --num-workers 0
uv run python train.py --model deit_small_patch16_224.fb_in1k           --optimizer sgd --lr 1e-2 --train-fraction 0.25 --num-workers 0
uv run python train.py --model deit_small_patch16_224.fb_in1k           --optimizer sgd --lr 1e-2 --train-fraction 0.1  --num-workers 0

uv run python train.py --model deit_small_distilled_patch16_224.fb_in1k --optimizer sgd --lr 1e-2 --train-fraction 0.5  --num-workers 0
uv run python train.py --model deit_small_distilled_patch16_224.fb_in1k --optimizer sgd --lr 1e-2 --train-fraction 0.25 --num-workers 0
uv run python train.py --model deit_small_distilled_patch16_224.fb_in1k --optimizer sgd --lr 1e-2 --train-fraction 0.1  --num-workers 0
```

> 9 Läufe (3 Modelle * 3 Datenanteile)
> Ergebnis-JSONs -> `results/stage2/`

## 6. Stufe 3 - Robustheit

Reine Inferenz auf den 3 Stufe-1-Checkpoints (Seed 42), kein erneutes Training.

### 6.1 Auswertung auf gestörten Testbildern - `scripts/evaluate_robustness.py`

- wertet jeden gefundenen Checkpoint auf dem kompletten Testset aus: ungestört ("clean") sowie 10 Störungen (Noise σ=0.1, Rotation 15°/30°/45°, Blur k=3/5/7, Occlusion 10 %/25 %/40 %)
- schreibt pro Modell eine Metrik-JSON sowie pro Modell * Störung eine CSV mit den Einzelvorhersagen (Pfad, wahres Label, Vorhersage), wird für `compare_models.py` benötigt

| Flag | Default | Bedeutung |
| --- | --- | --- |
| `--checkpoint-dir` | `results/stage1_checkpoints` | Verzeichnis mit den `*_best.pt`-Checkpoints |
| `--output-dir` | `outputs/eval` | Zielverzeichnis für JSON + `predictions/*.csv` |
| `--models` | alle gefundenen | nur bestimmte timm-Modellnamen auswerten |
| `--only` | alle 11 | nur bestimmte Störungen auswerten |
| `--batch-size` | `64` | Batch-Größe |
| `--num-workers` | `4` | Dataloader-Worker |
| `--seed` | `42` | Seed für Rauschen/Occlusion-Positionierung |
| `--limit-batches` | alle | nur die ersten N Batches (lokaler Smoke-Test) |

```bash
# voller Lauf, Ergebnisse direkt nach results/stage3/ schreiben
uv run python scripts/evaluate_robustness.py --output-dir results/stage3

# lokaler Smoke-Test (CPU, wenige Batches, ein Modell/eine Störung)
uv run python scripts/evaluate_robustness.py --models convnext_tiny.fb_in1k --only clean rotation15 --limit-batches 3
```

### 6.2 Statistischer Modellvergleich - `scripts/compare_models.py`

Liest die `clean`-Vorhersagen aller 3 Modelle aus `results/stage3/predictions/` und führt für jedes Modellpaar einen exakten Binomialtest (McNemar-äquivalent) auf den gepaarten Vorhersagen durch und beantwortet, ob ein Accuracy-Unterschied statistisch echt ist oder nur Zufall. Keine Flags, feste Modell- und Pfad-Liste im Skript.

```bash
uv run python scripts/compare_models.py
```

### 6.3 Beispiel-Abbildung - `scripts/plot_perturbations.py`

Rendert ein einzelnes Testbild mit allen 11 Störungen nebeneinander (dieselbe Störinstanz/Seed wie in der Evaluation), Ausgabe als PNG + PDF.

| Flag | Default | Bedeutung |
| --- | --- | --- |
| `--index` | `0` | welche Zeile aus `test.csv` verwendet wird |
| `--seed` | `42` | muss zum Evaluationslauf passen |

```bash
uv run python scripts/plot_perturbations.py
uv run python scripts/plot_perturbations.py --index 123   # anderes Beispielbild
```

-> `results/figures/stage3_perturbation_examples.png`/`.pdf`

## 7. Interpretierbarkeit - `scripts/interpretability.py`

Grad-CAM (ConvNeXt) und Attention Rollout (DeiT/DeiT-distilled) auf den Stufe-1-Checkpoints, reine Inferenz (+ 1 Backward-Pass für Grad-CAM).

| Flag | Default | Bedeutung |
| --- | --- | --- |
| `--model` | *(Pflicht)* | timm-Modellname |
| `--checkpoint` | *(Pflicht)* | Pfad zu einem `*_best.pt`-Checkpoint |
| `--images-dir` | keins | eigene Fotos statt EuroSAT-Testbildern (Domain-Shift-Demo) |
| `--num-images` | `6` | Anzahl Bilder (bei EuroSAT: ein Bild pro Klasse bis zum Limit) |
| `--splits-dir` | `data/splits` | Verzeichnis mit den Split-CSVs |
| `--out-dir` | `results/figures/interpretability` | Ausgabeverzeichnis für die Heatmap-PNGs |
| `--seed` | `42` | Seed für die Testbild-Auswahl |

```bash
uv run python scripts/interpretability.py \
    --model convnext_tiny.fb_in1k \
    --checkpoint results/stage1_checkpoints/convnext_tiny.fb_in1k_sgd_stage1_best.pt \
    --num-images 10

# eigene Fotos statt EuroSAT-Testbildern
uv run python scripts/interpretability.py \
    --model deit_small_distilled_patch16_224.fb_in1k \
    --checkpoint results/stage1_checkpoints/deit_small_distilled_patch16_224.fb_in1k_sgd_stage1_best.pt \
    --images-dir demo_images
```
