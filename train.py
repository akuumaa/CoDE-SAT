# TODO:
# - save split properly so all models use same data
# - add distilled deit later
# - save best model checkpoint
# - add data efficiency later
# - add robustness tests later

import argparse
import json
import time
from pathlib import Path

import timm
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm


def get_dataloaders(batch_size: int):
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

    try:
        dataset = datasets.EuroSAT(
            root="data",
            download=False,
            transform=transform,
        )
    except RuntimeError as error:
        raise RuntimeError(
            "EuroSAT was not found. Run prepare_eurosat.py"
        ) from error

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader, dataset.classes


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

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device: {device}")
    print(f"model: {args.model}")

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        batch_size=args.batch_size,
    )

    model = create_model(
        model_name=args.model,
        num_classes=len(classes),
    )

    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr)

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
        "train_time_seconds": train_time,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "test_f1": test_f1,
        "history": history,
    }

    Path("results/metrics").mkdir(parents=True, exist_ok=True)

    result_path = f"results/metrics/{args.model}_results.json"

    with open(result_path, "w") as file:
        json.dump(results, file, indent=4)

    print("\nfinal test result")
    print(f"test loss: {test_loss:.4f}")
    print(f"test acc:  {test_acc:.4f}")
    print(f"test f1:   {test_f1:.4f}")
    print(f"saved to:  {result_path}")


if __name__ == "__main__":
    main()