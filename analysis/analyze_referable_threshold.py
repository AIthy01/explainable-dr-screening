import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)

PRED_DIR = "results/predictions"
OUT_DIR = "results/referral"

W_VGG = 0.60
W_RESNET = 0.40

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Load
# ============================================================

labels = np.load(
    os.path.join(PRED_DIR, "labels.npy")
)

vgg_probs = np.load(
    os.path.join(PRED_DIR, "vgg16_probs.npy")
)

resnet_probs = np.load(
    os.path.join(PRED_DIR, "resnet50_probs.npy")
)

# ============================================================
# Fixed ensemble
# ============================================================

fused_probs = (
    W_VGG * vgg_probs
    + W_RESNET * resnet_probs
)

# Referable DR:
# grade 0/1 = negative
# grade 2/3/4 = positive
y_true = (labels >= 2).astype(int)

# IMPORTANT:
# This is treated as a screening SCORE,
# not yet as a calibrated clinical probability.
ref_score = fused_probs[:, 2:].sum(axis=1)

roc_auc = roc_auc_score(
    y_true,
    ref_score,
)

pr_auc = average_precision_score(
    y_true,
    ref_score,
)

# ============================================================
# Threshold sweep
# ============================================================

thresholds = np.arange(
    0.01,
    1.00,
    0.01,
)

rows = []

for t in thresholds:

    pred = (ref_score >= t).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = recall_score(
        y_true,
        pred,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    precision = precision_score(
        y_true,
        pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        pred,
        zero_division=0,
    )

    npv = (
        tn / (tn + fn)
        if (tn + fn) > 0
        else 0.0
    )

    youden_j = (
        sensitivity
        + specificity
        - 1
    )

    rows.append({
        "threshold": round(float(t), 2),
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision_ppv": precision,
        "npv": npv,
        "f1": f1,
        "youden_j": youden_j,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    })

df = pd.DataFrame(rows)

# ============================================================
# Operating points
# ============================================================

def nearest_threshold(target):
    idx = (
        df["threshold"] - target
    ).abs().idxmin()

    return df.loc[idx]


baseline_05 = nearest_threshold(0.50)

best_f1 = (
    df.sort_values(
        ["f1", "sensitivity_recall"],
        ascending=[False, False],
    )
    .iloc[0]
)

best_youden = (
    df.sort_values(
        ["youden_j", "sensitivity_recall"],
        ascending=[False, False],
    )
    .iloc[0]
)


def best_with_min_recall(min_recall):

    candidates = df[
        df["sensitivity_recall"]
        >= min_recall
    ].copy()

    if len(candidates) == 0:
        return None

    return (
        candidates.sort_values(
            [
                "specificity",
                "precision_ppv",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .iloc[0]
    )


recall_95 = best_with_min_recall(
    0.95
)

recall_97 = best_with_min_recall(
    0.97
)

# ============================================================
# Save
# ============================================================

df.to_csv(
    os.path.join(
        OUT_DIR,
        "referable_threshold_sweep.csv",
    ),
    index=False,
)

np.save(
    os.path.join(
        OUT_DIR,
        "referable_scores.npy",
    ),
    ref_score,
)

# ============================================================
# Print
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 180)

print("=" * 90)
print("REFERABLE DR SCORE ANALYSIS")
print("=" * 90)
print(f"Fusion weights : VGG16={W_VGG:.2f}, ResNet50={W_RESNET:.2f}")
print(f"Positive cases : {int(y_true.sum())}")
print(f"Negative cases : {int((1-y_true).sum())}")
print(f"ROC-AUC        : {roc_auc:.4f}")
print(f"PR-AUC         : {pr_auc:.4f}")


def print_point(title, row):

    print("\n" + "-" * 90)
    print(title)
    print("-" * 90)

    if row is None:
        print("No threshold satisfies this requirement.")
        return

    cols = [
        "threshold",
        "sensitivity_recall",
        "specificity",
        "precision_ppv",
        "npv",
        "f1",
        "youden_j",
        "TP",
        "FP",
        "FN",
        "TN",
    ]

    print(
        row[cols]
        .round(4)
        .to_string()
    )


print_point(
    "Baseline threshold t = 0.50",
    baseline_05,
)

print_point(
    "Threshold with maximum F1",
    best_f1,
)

print_point(
    "Threshold with maximum Youden J",
    best_youden,
)

print_point(
    "Highest specificity subject to Recall >= 95%",
    recall_95,
)

print_point(
    "Highest specificity subject to Recall >= 97%",
    recall_97,
)

print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)
print(
    os.path.join(
        OUT_DIR,
        "referable_threshold_sweep.csv",
    )
)
print(
    os.path.join(
        OUT_DIR,
        "referable_scores.npy",
    )
)
