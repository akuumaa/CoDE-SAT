# TODO:
# - add distilled deit later
# - add data efficiency later
# - add robustness tests later

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import SGD, AdamW, RMSprop
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm


class EuroSATSplit(Dataset):
    def __init__(self, csv_path: Path, transform):
        self.frame = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]

        image = Image.open(row["path"]).convert("RGB")
        image = self.transform(image)

        label = int(row["label"])

        return image, label


def load_classes(splits_dir: Path):
    classes_path = splits_dir / "classes.txt"

    classes_df = pd.read_csv(classes_path)
    classes_df = classes_df.sort_values("label")

    return classes_df["class_name"].tolist()


def get_dataloaders(batch_size: int, num_workers: int, splits_dir: Path = Path("data/splits")):
    transform = transforms.Compose([
        # pretrained timm models usually expect 224x224 inputs
        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        # imagenet normalization because we use pretrained weights
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    train_csv = splits_dir / "train.csv"
    val_csv = splits_dir / "val.csv"
    test_csv = splits_dir / "test.csv"

    for csv_path in (train_csv, val_csv, test_csv):
        if not csv_path.exists():
            raise FileNotFoundError(
                f"{csv_path} not found. Run scripts/prepare_eurosat.py "
                "and then scripts/make_splits.py."
            )

    train_set = EuroSATSplit(train_csv, transform)
    val_set = EuroSATSplit(val_csv, transform)
    test_set = EuroSATSplit(test_csv, transform)

    classes = load_classes(splits_dir)

    # pin_memory only helps when batches are copied to a GPU
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, classes


def create_optimizer(name: str, parameters, lr: float, momentum: float):
    if name == "adamw":
        return AdamW(parameters, lr=lr)

    if name == "sgd":
        return SGD(parameters, lr=lr, momentum=momentum)

    if name == "rmsprop":
        return RMSprop(parameters, lr=lr)

    raise ValueError(f"unknown optimizer: {name}")


def create_model(model_name: str, num_classes: int):
    model = timm.create_model(
        model_name,
        pretrained=True,
        num_classes=num_classes,
    )

    return model


def train_one_epoch(model, train_loader, loss_fn, optimizer, device):
    model.train()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in tqdm(train_loader, desc="train"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="macro")

    return avg_loss, accuracy, f1


@torch.no_grad()
def evaluate(model, data_loader, loss_fn, device):
    model.eval()

    total_loss = 0.0
    all_predictions = []
    all_labels = []

    for images, labels in tqdm(data_loader, desc="eval"):
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = loss_fn(outputs, labels)

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average="macro")

    return avg_loss, accuracy, f1


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        default="deit_tiny_patch16_224",
        help="timm model name",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="number of training epochs",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="batch size",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="number of dataloader worker processes",
    )

    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "sgd", "rmsprop"],
        help="optimizer for training",
    )

    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="momentum (only used by sgd)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device: {device}")
    print(f"model: {args.model}")
    print(f"optimizer: {args.optimizer}")
    print(f"seed: {args.seed}")

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = create_model(
        model_name=args.model,
        num_classes=len(classes),
    )

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = create_optimizer(
        name=args.optimizer,
        parameters=model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
    )

    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # optimizer in the filename so runs with different optimizers don't overwrite each other
    run_name = f"{args.model}_{args.optimizer}"
    checkpoint_path = checkpoint_dir / f"{run_name}_best.pt"
    best_val_acc = -1.0

    history = []
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nepoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        val_loss, val_acc, val_f1 = evaluate(
            model=model,
            data_loader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        print(f"train loss: {train_loss:.4f}")
        print(f"train acc:  {train_acc:.4f}")
        print(f"train f1:   {train_f1:.4f}")
        print(f"val loss:   {val_loss:.4f}")
        print(f"val acc:    {val_acc:.4f}")
        print(f"val f1:     {val_f1:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            print(f"saved new best checkpoint (val acc {val_acc:.4f})")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
        })

    train_time = time.time() - start_time

    # evaluate the best checkpoint, not just whatever the last epoch left behind
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    test_loss, test_acc, test_f1 = evaluate(
        model=model,
        data_loader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    results = {
        "model": args.model,
        "classes": classes,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "optimizer": args.optimizer,
        "momentum": args.momentum if args.optimizer == "sgd" else None,
        "seed": args.seed,
        "train_time_seconds": train_time,
        "best_val_acc": best_val_acc,
        "checkpoint_path": str(checkpoint_path),
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "history": history,
    }

    metrics_dir = Path("outputs/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    result_path = metrics_dir / f"{run_name}_results.json"

    with open(result_path, "w") as file:
        json.dump(results, file, indent=4)

    print("\nfinal test result (best checkpoint)")
    print(f"test loss: {test_loss:.4f}")
    print(f"test acc:  {test_acc:.4f}")
    print(f"test f1:   {test_f1:.4f}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"saved to:   {result_path}")


if __name__ == "__main__":
    main()