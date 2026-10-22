import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    confusion_matrix,
)

# ============================================================
# Configuration
# ============================================================

MODEL_LIST = [
    "vgg16",
    "resnet50",
    "densenet121",
    "efficientnet_b3",
]

NUM_CLASSES = 5

PRED_DIR = "results/predictions"
OUT_DIR = "results/analysis"

# 当前实验分析中暂时定义：
# DR 3 / 4 = high-risk
HIGH_RISK_THRESHOLD = 3

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Load data
# ============================================================

labels_path = os.path.join(PRED_DIR, "labels.npy")

if not os.path.exists(labels_path):
    raise FileNotFoundError(f"Missing: {labels_path}")

labels = np.load(labels_path)

predictions = {}
probabilities = {}

for model_name in MODEL_LIST:
    pred_path = os.path.join(
        PRED_DIR,
        f"{model_name}_preds.npy"
    )

    prob_path = os.path.join(
        PRED_DIR,
        f"{model_name}_probs.npy"
    )

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"Missing: {pred_path}")

    if not os.path.exists(prob_path):
        raise FileNotFoundError(f"Missing: {prob_path}")

    preds = np.load(pred_path)
    probs = np.load(prob_path)

    if len(preds) != len(labels):
        raise ValueError(
            f"{model_name}: preds length {len(preds)} "
            f"!= labels length {len(labels)}"
        )

    if probs.shape != (len(labels), NUM_CLASSES):
        raise ValueError(
            f"{model_name}: unexpected probs shape {probs.shape}"
        )

    predictions[model_name] = preds
    probabilities[model_name] = probs


print("=" * 70)
print("DATA CHECK")
print("=" * 70)
print(f"Samples: {len(labels)}")
print(f"Models : {MODEL_LIST}")


# ============================================================
# 1. Per-class metrics
# ============================================================

per_class_rows = []
model_summary_rows = []
high_risk_rows = []

for model_name in MODEL_LIST:

    preds = predictions[model_name]

    precision, recall, f1, support = (
        precision_recall_fscore_support(
            labels,
            preds,
            labels=list(range(NUM_CLASSES)),
            zero_division=0,
        )
    )

    for cls in range(NUM_CLASSES):
        per_class_rows.append(
            {
                "model": model_name,
                "class": cls,
                "precision": precision[cls],
                "recall": recall[cls],
                "f1": f1[cls],
                "support": support[cls],
            }
        )

    accuracy = accuracy_score(labels, preds)

    macro_f1 = f1_score(
        labels,
        preds,
        average="macro",
        zero_division=0,
    )

    # --------------------------------
    # Low / medium group: true class 0/1/2
    # --------------------------------

    low_mid_mask = labels < HIGH_RISK_THRESHOLD

    low_mid_macro_f1 = f1_score(
        labels[low_mid_mask],
        preds[low_mid_mask],
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )

    # --------------------------------
    # High-risk binary analysis
    # true high-risk = label >= 3
    # predicted high-risk = prediction >= 3
    # --------------------------------

    y_true_high = (
        labels >= HIGH_RISK_THRESHOLD
    ).astype(int)

    y_pred_high = (
        preds >= HIGH_RISK_THRESHOLD
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_high,
        y_pred_high,
        labels=[0, 1],
    ).ravel()

    high_precision = precision_score(
        y_true_high,
        y_pred_high,
        zero_division=0,
    )

    high_recall = recall_score(
        y_true_high,
        y_pred_high,
        zero_division=0,
    )

    high_f1 = f1_score(
        y_true_high,
        y_pred_high,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    high_risk_rows.append(
        {
            "model": model_name,
            "high_risk_precision": high_precision,
            "high_risk_recall": high_recall,
            "high_risk_specificity": specificity,
            "high_risk_f1": high_f1,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        }
    )

    model_summary_rows.append(
        {
            "model": model_name,
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "low_mid_macro_f1": low_mid_macro_f1,
            "high_risk_recall": high_recall,
            "high_risk_precision": high_precision,
            "high_risk_specificity": specificity,
            "high_risk_f1": high_f1,
            "high_risk_FN": fn,
        }
    )


per_class_df = pd.DataFrame(per_class_rows)

high_risk_df = pd.DataFrame(high_risk_rows)

model_summary_df = pd.DataFrame(
    model_summary_rows
)


# ============================================================
# 2. Pairwise complementarity
# ============================================================

pair_rows = []
rescue_rows = []

for i in range(len(MODEL_LIST)):
    for j in range(i + 1, len(MODEL_LIST)):

        model_a = MODEL_LIST[i]
        model_b = MODEL_LIST[j]

        pred_a = predictions[model_a]
        pred_b = predictions[model_b]

        correct_a = pred_a == labels
        correct_b = pred_b == labels

        error_a = ~correct_a
        error_b = ~correct_b

        both_wrong = np.logical_and(
            error_a,
            error_b,
        )

        either_wrong = np.logical_or(
            error_a,
            error_b,
        )

        a_wrong_b_right = np.logical_and(
            error_a,
            correct_b,
        )

        a_right_b_wrong = np.logical_and(
            correct_a,
            error_b,
        )

        both_right = np.logical_and(
            correct_a,
            correct_b,
        )

        overlap = (
            both_wrong.sum() / either_wrong.sum()
            if either_wrong.sum() > 0
            else 0
        )

        # 理论“oracle”：
        # 只要两模型中至少一个预测正确，就认为能选到正确答案。
        oracle_accuracy = (
            1.0 - both_wrong.mean()
        )

        high_mask = (
            labels >= HIGH_RISK_THRESHOLD
        )

        pair_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "error_overlap": overlap,
                "both_right": both_right.sum(),
                "both_wrong": both_wrong.sum(),
                "a_wrong_b_right": a_wrong_b_right.sum(),
                "a_right_b_wrong": a_right_b_wrong.sum(),
                "oracle_accuracy": oracle_accuracy,
                "high_risk_a_wrong_b_right":
                    np.logical_and(
                        a_wrong_b_right,
                        high_mask,
                    ).sum(),
                "high_risk_a_right_b_wrong":
                    np.logical_and(
                        a_right_b_wrong,
                        high_mask,
                    ).sum(),
                "high_risk_both_wrong":
                    np.logical_and(
                        both_wrong,
                        high_mask,
                    ).sum(),
            }
        )

        # --------------------------------
        # Rescue analysis by true class
        # --------------------------------

        for cls in range(NUM_CLASSES):

            cls_mask = labels == cls

            rescue_rows.append(
                {
                    "model_a": model_a,
                    "model_b": model_b,
                    "true_class": cls,
                    "support": cls_mask.sum(),
                    "a_wrong_b_right":
                        np.logical_and(
                            a_wrong_b_right,
                            cls_mask,
                        ).sum(),
                    "a_right_b_wrong":
                        np.logical_and(
                            a_right_b_wrong,
                            cls_mask,
                        ).sum(),
                    "both_wrong":
                        np.logical_and(
                            both_wrong,
                            cls_mask,
                        ).sum(),
                    "both_right":
                        np.logical_and(
                            both_right,
                            cls_mask,
                        ).sum(),
                }
            )


pair_df = pd.DataFrame(pair_rows)

rescue_df = pd.DataFrame(rescue_rows)


# ============================================================
# Save
# ============================================================

per_class_df.to_csv(
    os.path.join(
        OUT_DIR,
        "per_class_metrics.csv",
    ),
    index=False,
)

high_risk_df.to_csv(
    os.path.join(
        OUT_DIR,
        "high_risk_metrics.csv",
    ),
    index=False,
)

model_summary_df.to_csv(
    os.path.join(
        OUT_DIR,
        "model_summary.csv",
    ),
    index=False,
)

pair_df.to_csv(
    os.path.join(
        OUT_DIR,
        "pairwise_complementarity.csv",
    ),
    index=False,
)

rescue_df.to_csv(
    os.path.join(
        OUT_DIR,
        "pairwise_rescue_by_class.csv",
    ),
    index=False,
)


# ============================================================
# Print key results
# ============================================================

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("\n")
print("=" * 70)
print("MODEL SUMMARY")
print("=" * 70)

print(
    model_summary_df.round(4).to_string(
        index=False
    )
)

print("\n")
print("=" * 70)
print("PER-CLASS METRICS")
print("=" * 70)

print(
    per_class_df.round(4).to_string(
        index=False
    )
)

print("\n")
print("=" * 70)
print("HIGH-RISK ANALYSIS (DR 3 + 4)")
print("=" * 70)

print(
    high_risk_df.round(4).to_string(
        index=False
    )
)

print("\n")
print("=" * 70)
print("PAIRWISE COMPLEMENTARITY")
print("=" * 70)

pair_print = pair_df.sort_values(
    "error_overlap"
)

print(
    pair_print.round(4).to_string(
        index=False
    )
)

print("\n")
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

for filename in [
    "model_summary.csv",
    "per_class_metrics.csv",
    "high_risk_metrics.csv",
    "pairwise_complementarity.csv",
    "pairwise_rescue_by_class.csv",
]:
    print(
        os.path.join(
            OUT_DIR,
            filename,
        )
    )
