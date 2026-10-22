import os
import random
import argparse
import torch
import numpy as np
import pandas as pd

from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from dataset import DRDataset
from transforms import get_train_transform, get_val_transform
from model import get_model
from config import *


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=MODEL_NAME,
        choices=["vgg16", "resnet50", "densenet121", "efficientnet_b3"]
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS
    )
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_class_weights(labels):
    counts = labels.value_counts().reindex(
        range(NUM_CLASSES),
        fill_value=0
    )

    if (counts == 0).any():
        raise ValueError(
            f"训练集中存在缺失类别，类别计数为: {counts.to_dict()}"
        )

    weights = len(labels) / (NUM_CLASSES * counts.values)

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=DEVICE
    )


def evaluate(model, loader, criterion):
    model.eval()

    running_loss = 0.0
    total_samples = 0

    preds = []
    labels = []
    probs = []

    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(
                DEVICE,
                non_blocking=True
            )

            targets = targets.to(
                DEVICE,
                non_blocking=True
            )

            outputs = model(images)
            loss = criterion(outputs, targets)

            batch_size = images.size(0)

            running_loss += loss.item() * batch_size
            total_samples += batch_size

            probability = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)

            preds.extend(
                prediction.cpu().numpy()
            )

            labels.extend(
                targets.cpu().numpy()
            )

            probs.extend(
                probability.cpu().numpy()
            )

    val_loss = running_loss / total_samples

    acc = accuracy_score(
        labels,
        preds
    )

    f1 = f1_score(
        labels,
        preds,
        average="macro",
        zero_division=0
    )

    try:
        auc = roc_auc_score(
            labels,
            probs,
            multi_class="ovr",
            average="macro"
        )
    except ValueError:
        auc = np.nan

    return val_loss, acc, f1, auc


def main():
    args = parse_args()

    model_name = args.model
    epochs = args.epochs

    save_path = (
        f"results/checkpoints/"
        f"{model_name}_best.pth"
    )

    set_seed(42)

    df = pd.read_csv(TRAIN_CSV)

    indices = np.arange(
        len(df)
    )

    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=df["diagnosis"]
    )

    train_labels = df.iloc[
        train_idx
    ]["diagnosis"]

    class_weights = get_class_weights(
        train_labels
    )

    print(f"Device: {DEVICE}")
    print(f"Training model: {model_name}")
    print(f"Total samples: {len(df)}")
    print(f"Train samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")

    print(
        "Training class distribution:",
        train_labels.value_counts()
        .sort_index()
        .to_dict()
    )

    print(
        "Class weights:",
        class_weights.detach()
        .cpu()
        .numpy()
        .round(4)
        .tolist()
    )

    train_dataset = Subset(
        DRDataset(
            TRAIN_CSV,
            IMAGE_DIR,
            get_train_transform()
        ),
        train_idx
    )

    val_dataset = Subset(
        DRDataset(
            TRAIN_CSV,
            IMAGE_DIR,
            get_val_transform()
        ),
        val_idx
    )

    pin_memory = (
        DEVICE == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4
)

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        persistent_workers=True,
        prefetch_factor=4
)

    model = get_model(
        model_name
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    best_f1 = -float("inf")

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1": [],
        "val_auc": [],
        "lr": []
    }

    os.makedirs(
        "results/checkpoints",
        exist_ok=True
    )

    os.makedirs(
        "results/metrics",
        exist_ok=True
    )

    for epoch in range(epochs):
        model.train()

        running_loss = 0.0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()
            optimizer.step()

            batch_size = images.size(0)

            running_loss += (
                loss.item() * batch_size
            )

            total_samples += batch_size

        train_loss = (
            running_loss /
            total_samples
        )

        val_loss, val_acc, val_f1, val_auc = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(
            val_f1
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["epoch"].append(
            epoch + 1
        )

        history["train_loss"].append(
            train_loss
        )

        history["val_loss"].append(
            val_loss
        )

        history["val_acc"].append(
            val_acc
        )

        history["val_f1"].append(
            val_f1
        )

        history["val_auc"].append(
            val_auc
        )

        history["lr"].append(
            current_lr
        )

        print(
            f"Epoch [{epoch + 1}/{epochs}] "
            f"TrainLoss:{train_loss:.4f} "
            f"ValLoss:{val_loss:.4f} "
            f"Acc:{val_acc:.4f} "
            f"F1:{val_f1:.4f} "
            f"AUC:{val_auc:.4f} "
            f"LR:{current_lr:.2e}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1

            torch.save(
                model.state_dict(),
                save_path
            )

            print(
                f"Best model saved: "
                f"{save_path}"
            )

        pd.DataFrame(
            history
        ).to_csv(
            f"results/metrics/"
            f"{model_name}_history.csv",
            index=False
        )

    print("Training finished.")
    print(
        f"Best Macro-F1: "
        f"{best_f1:.4f}"
    )


if __name__ == "__main__":
    main()