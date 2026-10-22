import os
import itertools
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
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

PRED_DIR = "results/predictions"
OUT_DIR = "results/ensemble"

WEIGHTS = np.arange(0.0, 1.0001, 0.05)

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Helper functions
# ============================================================

def binary_metrics(labels, preds, threshold):
    """
    将五分类标签/预测按 grade >= threshold 转为二分类，
    返回 Precision / Recall / Specificity / F1 / TP / FP / FN / TN。
    """

    y_true = (labels >= threshold).astype(int)
    y_pred = (preds >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


def choose_best(df, metric, extra_sort=None):
    """
    按指定指标选最佳结果。
    如主指标并列，可通过 extra_sort 提供次级排序规则。
    """

    sort_cols = [metric]
    ascending = [False]

    if extra_sort:
        for col, asc in extra_sort:
            sort_cols.append(col)
            ascending.append(asc)

    return (
        df.sort_values(
            sort_cols,
            ascending=ascending,
        )
        .iloc[0]
    )


# ============================================================
# Load saved probabilities
# ============================================================

labels_path = os.path.join(
    PRED_DIR,
    "labels.npy",
)

if not os.path.exists(labels_path):
    raise FileNotFoundError(
        f"Missing: {labels_path}"
    )

labels = np.load(labels_path)

probabilities = {}

for model in MODEL_LIST:

    path = os.path.join(
        PRED_DIR,
        f"{model}_probs.npy",
    )

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {path}"
        )

    probs = np.load(path)

    if probs.shape != (len(labels), 5):
        raise ValueError(
            f"{model}: unexpected probability shape "
            f"{probs.shape}, expected ({len(labels)}, 5)"
        )

    probabilities[model] = probs


print("=" * 90)
print("WEIGHTED SOFT VOTING SEARCH")
print("=" * 90)
print(f"Samples       : {len(labels)}")
print(f"Models        : {MODEL_LIST}")
print(f"Weight values : {len(WEIGHTS)}")
print(f"Model pairs   : {len(list(itertools.combinations(MODEL_LIST, 2)))}")


# ============================================================
# Search all pairs and weights
# ============================================================

rows = []

for model_a, model_b in itertools.combinations(
    MODEL_LIST,
    2,
):

    probs_a = probabilities[model_a]
    probs_b = probabilities[model_b]

    print(
        f"\nSearching: {model_a} + {model_b}"
    )

    for w in WEIGHTS:

        # w belongs to model_a
        # (1-w) belongs to model_b
        fused_probs = (
            w * probs_a
            + (1.0 - w) * probs_b
        )

        fused_preds = np.argmax(
            fused_probs,
            axis=1,
        )

        # ----------------------------------------------------
        # Five-class metrics
        # ----------------------------------------------------

        accuracy = accuracy_score(
            labels,
            fused_preds,
        )

        macro_f1 = f1_score(
            labels,
            fused_preds,
            average="macro",
            zero_division=0,
        )

        try:
            auc = roc_auc_score(
                labels,
                fused_probs,
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            auc = np.nan

        # ----------------------------------------------------
        # Referable DR: grade >= 2
        # ----------------------------------------------------

        referable = binary_metrics(
            labels,
            fused_preds,
            threshold=2,
        )

        # ----------------------------------------------------
        # Severe-or-worse DR: grade >= 3
        # ----------------------------------------------------

        severe = binary_metrics(
            labels,
            fused_preds,
            threshold=3,
        )

        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,

                "weight_a": round(
                    float(w),
                    2,
                ),
                "weight_b": round(
                    float(1.0 - w),
                    2,
                ),

                # Five-class
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "auc": auc,

                # Referable DR >= 2
                "referable_precision":
                    referable["precision"],
                "referable_recall":
                    referable["recall"],
                "referable_specificity":
                    referable["specificity"],
                "referable_f1":
                    referable["f1"],
                "referable_TP":
                    referable["TP"],
                "referable_FP":
                    referable["FP"],
                "referable_FN":
                    referable["FN"],
                "referable_TN":
                    referable["TN"],

                # Severe-or-worse >= 3
                "severe_precision":
                    severe["precision"],
                "severe_recall":
                    severe["recall"],
                "severe_specificity":
                    severe["specificity"],
                "severe_f1":
                    severe["f1"],
                "severe_TP":
                    severe["TP"],
                "severe_FP":
                    severe["FP"],
                "severe_FN":
                    severe["FN"],
                "severe_TN":
                    severe["TN"],
            }
        )


results_df = pd.DataFrame(rows)

search_path = os.path.join(
    OUT_DIR,
    "ensemble_weight_search.csv",
)

results_df.to_csv(
    search_path,
    index=False,
)


# ============================================================
# Best result for each pair
# ============================================================

best_rows = []

for model_a, model_b in itertools.combinations(
    MODEL_LIST,
    2,
):

    pair_df = results_df[
        (results_df["model_a"] == model_a)
        & (results_df["model_b"] == model_b)
    ].copy()

    # --------------------------------------------------------
    # Best five-class Macro-F1
    # tie break:
    # Accuracy -> AUC
    # --------------------------------------------------------

    best_macro = choose_best(
        pair_df,
        "macro_f1",
        extra_sort=[
            ("accuracy", False),
            ("auc", False),
        ],
    )

    best_rows.append(
        {
            "model_a": model_a,
            "model_b": model_b,
            "objective":
                "best_macro_f1",
            **best_macro.to_dict(),
        }
    )

    # --------------------------------------------------------
    # Best Referable F1
    # tie break:
    # Recall -> Specificity
    # --------------------------------------------------------

    best_ref_f1 = choose_best(
        pair_df,
        "referable_f1",
        extra_sort=[
            ("referable_recall", False),
            ("referable_specificity", False),
        ],
    )

    best_rows.append(
        {
            "model_a": model_a,
            "model_b": model_b,
            "objective":
                "best_referable_f1",
            **best_ref_f1.to_dict(),
        }
    )

    # --------------------------------------------------------
    # Highest Referable Recall
    # tie break:
    # fewer FP -> higher F1
    # --------------------------------------------------------

    best_ref_recall = choose_best(
        pair_df,
        "referable_recall",
        extra_sort=[
            ("referable_FP", True),
            ("referable_f1", False),
        ],
    )

    best_rows.append(
        {
            "model_a": model_a,
            "model_b": model_b,
            "objective":
                "best_referable_recall",
            **best_ref_recall.to_dict(),
        }
    )

    # --------------------------------------------------------
    # Best Severe-or-worse F1
    # --------------------------------------------------------

    best_severe_f1 = choose_best(
        pair_df,
        "severe_f1",
        extra_sort=[
            ("severe_recall", False),
            ("severe_specificity", False),
        ],
    )

    best_rows.append(
        {
            "model_a": model_a,
            "model_b": model_b,
            "objective":
                "best_severe_f1",
            **best_severe_f1.to_dict(),
        }
    )


best_df = pd.DataFrame(
    best_rows
)

# Remove duplicated model_a/model_b columns caused by dict expansion
best_df = best_df.loc[
    :,
    ~best_df.columns.duplicated()
]

best_path = os.path.join(
    OUT_DIR,
    "ensemble_best_by_objective.csv",
)

best_df.to_csv(
    best_path,
    index=False,
)


# ============================================================
# Global summaries
# ============================================================

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    260,
)

print("\n")
print("=" * 90)
print("BEST FIVE-CLASS MACRO-F1 FOR EACH PAIR")
print("=" * 90)

macro_table = (
    best_df[
        best_df["objective"]
        == "best_macro_f1"
    ][
        [
            "model_a",
            "model_b",
            "weight_a",
            "weight_b",
            "accuracy",
            "macro_f1",
            "auc",
            "referable_recall",
            "referable_f1",
            "severe_recall",
            "severe_f1",
        ]
    ]
    .sort_values(
        "macro_f1",
        ascending=False,
    )
)

print(
    macro_table
    .round(4)
    .to_string(index=False)
)


print("\n")
print("=" * 90)
print("BEST REFERABLE-DR F1 FOR EACH PAIR")
print("=" * 90)

ref_table = (
    best_df[
        best_df["objective"]
        == "best_referable_f1"
    ][
        [
            "model_a",
            "model_b",
            "weight_a",
            "weight_b",
            "macro_f1",
            "referable_precision",
            "referable_recall",
            "referable_specificity",
            "referable_f1",
            "referable_FN",
            "referable_FP",
        ]
    ]
    .sort_values(
        "referable_f1",
        ascending=False,
    )
)

print(
    ref_table
    .round(4)
    .to_string(index=False)
)


print("\n")
print("=" * 90)
print("HIGHEST REFERABLE-DR RECALL FOR EACH PAIR")
print("=" * 90)

recall_table = (
    best_df[
        best_df["objective"]
        == "best_referable_recall"
    ][
        [
            "model_a",
            "model_b",
            "weight_a",
            "weight_b",
            "macro_f1",
            "referable_precision",
            "referable_recall",
            "referable_specificity",
            "referable_f1",
            "referable_FN",
            "referable_FP",
        ]
    ]
    .sort_values(
        [
            "referable_recall",
            "referable_FP",
        ],
        ascending=[
            False,
            True,
        ],
    )
)

print(
    recall_table
    .round(4)
    .to_string(index=False)
)


print("\n")
print("=" * 90)
print("BEST SEVERE-OR-WORSE F1 FOR EACH PAIR")
print("=" * 90)

severe_table = (
    best_df[
        best_df["objective"]
        == "best_severe_f1"
    ][
        [
            "model_a",
            "model_b",
            "weight_a",
            "weight_b",
            "macro_f1",
            "severe_precision",
            "severe_recall",
            "severe_specificity",
            "severe_f1",
            "severe_FN",
            "severe_FP",
        ]
    ]
    .sort_values(
        "severe_f1",
        ascending=False,
    )
)

print(
    severe_table
    .round(4)
    .to_string(index=False)
)


print("\n")
print("=" * 90)
print("FILES SAVED")
print("=" * 90)
print(search_path)
print(best_path)
