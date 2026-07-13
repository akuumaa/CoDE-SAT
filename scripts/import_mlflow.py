"""
    import the result jsons under results/ into a local mlflow store

    - training ran on the pod without mlflow, this backfills the runs
      from the committed jsons
    - one run per json: params, per-epoch metrics (step = epoch),
      final metrics, the json itself as artifact
    - experiment name = parent folder (results/stage2 -> "stage2")
    - stage-3 robustness jsons have no history, they get acc/f1/loss
      per perturbation instead
    - safe to re-run, existing runs are skipped

    run:
        uv run python scripts/import_mlflow.py
    browse:
        uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

import json
from pathlib import Path

import mlflow

ROOT = Path(__file__).resolve().parent.parent

# sqlite backend, mlflow 3 deprecated the plain folder store
mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlflow.db'}")

# what gets taken over from the training jsons
PARAMS = ["model", "optimizer", "momentum", "scheduler", "learning_rate", "epochs",
          "batch_size", "seed", "train_fraction", "num_train_images", "tag", "num_params"]
FINAL = ["best_val_acc", "test_acc", "test_f1", "test_loss", "train_time_seconds"]


def set_experiment(name):
    # artifacts should land in mlruns/ inside the repo, no matter the cwd
    if mlflow.get_experiment_by_name(name) is None:
        mlflow.create_experiment(name, artifact_location=(ROOT / "mlruns" / name).as_uri())
    mlflow.set_experiment(name)


def run_exists(experiment, run_name):
    exp = mlflow.get_experiment_by_name(experiment)
    if exp is None:
        return False
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    return (not runs.empty) and run_name in list(runs.get("tags.mlflow.runName", []))


def import_training(json_path, experiment):
    run_name = json_path.stem.removesuffix("_results")
    set_experiment(experiment)
    if run_exists(experiment, run_name):
        print(f"skip (exists): {experiment}/{run_name}")
        return

    with open(json_path) as file:
        r = json.load(file)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: r[k] for k in PARAMS if r.get(k) is not None})

        # per-epoch curves (step = epoch), that is what the mlflow charts plot
        for h in r.get("history", []):
            for key, value in h.items():
                if key != "epoch":
                    mlflow.log_metric(key, value, step=h["epoch"])

        for key in FINAL:
            if key in r:
                mlflow.log_metric(key, r[key])

        if "inference" in r:
            mlflow.log_metric("ms_per_image", r["inference"]["ms_per_image"])
            mlflow.log_metric("images_per_second", r["inference"]["images_per_second"])

        mlflow.log_artifact(str(json_path))

    print(f"imported: {experiment}/{run_name}")


def import_robustness(json_path, experiment):
    run_name = json_path.stem
    set_experiment(experiment)
    if run_exists(experiment, run_name):
        print(f"skip (exists): {experiment}/{run_name}")
        return

    with open(json_path) as file:
        r = json.load(file)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_params({k: r[k] for k in ["model", "seed", "batch_size", "checkpoint"]})

        for pert_name, entry in r["results"].items():
            mlflow.log_metric(f"acc_{pert_name}", entry["acc"])
            mlflow.log_metric(f"f1_{pert_name}", entry["f1"])
            mlflow.log_metric(f"loss_{pert_name}", entry["loss"])

        mlflow.log_artifact(str(json_path))

    print(f"imported: {experiment}/{run_name}")


def main():
    for json_path in sorted((ROOT / "results").rglob("*.json")):
        if json_path.name.endswith("_stage3_robustness.json"):
            import_robustness(json_path, json_path.parent.name)
        elif json_path.name.endswith("_results.json"):
            import_training(json_path, json_path.parent.name)


if __name__ == "__main__":
    main()