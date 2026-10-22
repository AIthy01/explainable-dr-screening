import os
import torch
import numpy as np
import pandas as pd
import argparse
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

from dataset import DRDataset
from transforms import get_val_transform
from model import get_model
from config import *

MODEL_LIST = [
    "vgg16",
    "resnet50",
    "densenet121",
    "efficientnet_b3"
]

def evaluate_model(model, loader):
    model.eval()

    preds = []
    labels = []
    probs = []

    with torch.no_grad():
        for images, targets in loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)

            outputs = model(images)

            probability = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)

            preds.extend(prediction.cpu().numpy())
            labels.extend(targets.cpu().numpy())
            probs.extend(probability.cpu().numpy())

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    try:
        auc = roc_auc_score(labels, probs, multi_class="ovr")
    except:
        auc = 0

    cm = confusion_matrix(labels, preds)

    return acc, f1, auc, np.array(labels), np.array(preds), np.array(probs), cm

def get_val_loader():

    df = pd.read_csv(TRAIN_CSV)

    _, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        random_state=42,
        stratify=df["diagnosis"]
    )

    dataset = Subset(
        DRDataset(
            TRAIN_CSV,
            IMAGE_DIR,
            get_val_transform()
        ),
        val_idx
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    return loader

def error_overlap(pred1, pred2, labels):

    error1 = pred1 != labels
    error2 = pred2 != labels

    intersection = np.logical_and(error1, error2).sum()
    union = np.logical_or(error1, error2).sum()

    return intersection / union if union != 0 else 0

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate diabetic retinopathy models."
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--model",
        type=str,
        choices=MODEL_LIST,
        help="Evaluate one specific model."
    )

    group.add_argument(
        "--all",
        action="store_true",
        help="Evaluate all models and calculate error overlap."
    )

    args = parser.parse_args()

    if args.all:
        names = MODEL_LIST
    else:
        names = [args.model]

    os.makedirs("results/predictions", exist_ok=True)
    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures/confusion_matrix", exist_ok=True)

    loader = get_val_loader()

    metrics = []
    predictions = {}


    for name in names:

        print(f"Evaluating {name}...")

        model = get_model(name).to(DEVICE)

        weight_path = f"results/checkpoints/{name}_best.pth"

        model.load_state_dict(
            torch.load(weight_path, map_location=DEVICE)
        )


        acc, f1, auc, labels, preds, probs, cm = evaluate_model(
            model,
            loader
        )


        metrics.append(
            [name, acc, f1, auc]
        )


        predictions[name] = {
            "labels": labels,
            "preds": preds,
            "probs": probs
        }


        np.save(
            f"results/predictions/{name}_probs.npy",
            probs
        )

        np.save(
            f"results/predictions/{name}_preds.npy",
            preds
        )

        np.save(
            f"results/figures/confusion_matrix/{name}.npy",
            cm
        )

        np.save(
            "results/predictions/labels.npy",
            labels
        )


        print(
            f"{name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}"
        )


        metrics_df = pd.DataFrame(
            metrics,
            columns=["model", "accuracy", "f1", "auc"]
    )

    if args.all:
        metrics_path = "results/metrics/model_metrics.csv"
    else:
        metrics_path = f"results/metrics/{args.model}_metrics.csv"

    metrics_df.to_csv(
        metrics_path,
        index=False
    )

    print(f"\nMetrics saved to: {metrics_path}")


    if len(names) >= 2:
        print("\nError overlap:")

        for i in range(len(names)):
            for j in range(i + 1, len(names)):

                overlap = error_overlap(
                    predictions[names[i]]["preds"],
                    predictions[names[j]]["preds"],
                    predictions[names[i]]["labels"]
                )

                print(
                    f"{names[i]} + {names[j]} : {overlap:.4f}"
                )
    else:
        print("\nError overlap skipped: only one model evaluated.")


if __name__ == "__main__":
    main()