import os
import numpy as np
import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

MODEL_LIST = [
    "vgg16",
    "resnet50",
    "densenet121",
    "efficientnet_b3",
]

PRED_DIR = "results/predictions"
OUT_DIR = "results/analysis"

# 两个不同临床分析层次
TASKS = {
    "referable_dr_ge2": {
        "threshold": 2,
        "description": "Referable DR: grade >= 2",
    },
    "severe_or_worse_ge3": {
        "threshold": 3,
        "description": "Severe-or-worse DR: grade >= 3",
    },
}

os.makedirs(OUT_DIR, exist_ok=True)

labels = np.load(
    os.path.join(PRED_DIR, "labels.npy")
)

predictions = {}
probabilities = {}

for model in MODEL_LIST:
    predictions[model] = np.load(
        os.path.join(
            PRED_DIR,
            f"{model}_preds.npy",
        )
    )

    probabilities[model] = np.load(
        os.path.join(
            PRED_DIR,
            f"{model}_probs.npy",
        )
    )

print("=" * 90)
print("SCREENING THRESHOLD ANALYSIS")
print("=" * 90)
print(f"Samples: {len(labels)}")
print()


all_single_rows = []
all_pair_rows = []


for task_name, task_cfg in TASKS.items():

    threshold = task_cfg["threshold"]
    description = task_cfg["description"]

    true_positive_group = (
        labels >= threshold
    )

    y_true = true_positive_group.astype(int)

    print("\n" + "=" * 90)
    print(description)
    print("=" * 90)

    print(
        f"Positive samples: {true_positive_group.sum()}"
    )
    print(
        f"Negative samples: {(~true_positive_group).sum()}"
    )

    # ========================================================
    # 1. 单模型筛查性能
    # ========================================================

    single_rows = []

    for model in MODEL_LIST:

        preds = predictions[model]

        pred_positive = (
            preds >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            pred_positive,
            labels=[0, 1],
        ).ravel()

        precision = precision_score(
            y_true,
            pred_positive,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            pred_positive,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            pred_positive,
            zero_division=0,
        )

        specificity = (
            tn / (tn + fp)
            if (tn + fp) > 0
            else 0
        )

        row = {
            "task": task_name,
            "threshold": threshold,
            "model": model,
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "f1": f1,
            "TP": int(tp),
            "FP": int(fp),
            "FN": int(fn),
            "TN": int(tn),
        }

        single_rows.append(row)
        all_single_rows.append(row)

    single_df = pd.DataFrame(single_rows)

    print("\nSingle-model screening performance:")
    print(
        single_df[
            [
                "model",
                "precision",
                "recall",
                "specificity",
                "f1",
                "TP",
                "FP",
                "FN",
                "TN",
            ]
        ].round(4).to_string(index=False)
    )

    # ========================================================
    # 2. 两模型互补性
    # ========================================================

    pair_rows = []

    for i in range(len(MODEL_LIST)):
        for j in range(i + 1, len(MODEL_LIST)):

            model_a = MODEL_LIST[i]
            model_b = MODEL_LIST[j]

            pred_a_positive = (
                predictions[model_a] >= threshold
            )

            pred_b_positive = (
                predictions[model_b] >= threshold
            )

            # ------------------------------------------------
            # 真实阳性病例中，两个模型分别有没有筛出来
            # ------------------------------------------------

            both_detect = np.sum(
                true_positive_group
                & pred_a_positive
                & pred_b_positive
            )

            a_only_detect = np.sum(
                true_positive_group
                & pred_a_positive
                & ~pred_b_positive
            )

            b_only_detect = np.sum(
                true_positive_group
                & ~pred_a_positive
                & pred_b_positive
            )

            both_miss = np.sum(
                true_positive_group
                & ~pred_a_positive
                & ~pred_b_positive
            )

            # ------------------------------------------------
            # OR rule:
            # 任意一个模型认为需要筛出，就判为阳性
            # 用于观察理论互补价值，不作为最终集成方案
            # ------------------------------------------------

            pred_or = (
                pred_a_positive
                | pred_b_positive
            ).astype(int)

            tn, fp, fn, tp = confusion_matrix(
                y_true,
                pred_or,
                labels=[0, 1],
            ).ravel()

            or_precision = precision_score(
                y_true,
                pred_or,
                zero_division=0,
            )

            or_recall = recall_score(
                y_true,
                pred_or,
                zero_division=0,
            )

            or_f1 = f1_score(
                y_true,
                pred_or,
                zero_division=0,
            )

            or_specificity = (
                tn / (tn + fp)
                if (tn + fp) > 0
                else 0
            )

            row = {
                "task": task_name,
                "threshold": threshold,
                "model_a": model_a,
                "model_b": model_b,
                "positive_total":
                    int(true_positive_group.sum()),
                "both_detect":
                    int(both_detect),
                "a_only_detect":
                    int(a_only_detect),
                "b_only_detect":
                    int(b_only_detect),
                "both_miss":
                    int(both_miss),
                "or_precision":
                    or_precision,
                "or_recall":
                    or_recall,
                "or_specificity":
                    or_specificity,
                "or_f1":
                    or_f1,
                "or_TP":
                    int(tp),
                "or_FP":
                    int(fp),
                "or_FN":
                    int(fn),
                "or_TN":
                    int(tn),
            }

            pair_rows.append(row)
            all_pair_rows.append(row)

    pair_df = pd.DataFrame(pair_rows)

    pair_df = pair_df.sort_values(
        [
            "or_recall",
            "or_f1",
            "or_specificity",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    )

    print("\nPairwise screening complementarity:")
    print(
        pair_df[
            [
                "model_a",
                "model_b",
                "both_detect",
                "a_only_detect",
                "b_only_detect",
                "both_miss",
                "or_precision",
                "or_recall",
                "or_specificity",
                "or_f1",
                "or_FN",
                "or_FP",
            ]
        ].round(4).to_string(index=False)
    )


# ============================================================
# 保存总结果
# ============================================================

single_all_df = pd.DataFrame(
    all_single_rows
)

pair_all_df = pd.DataFrame(
    all_pair_rows
)

single_path = os.path.join(
    OUT_DIR,
    "screening_single_model_metrics.csv",
)

pair_path = os.path.join(
    OUT_DIR,
    "screening_pairwise_complementarity.csv",
)

single_all_df.to_csv(
    single_path,
    index=False,
)

pair_all_df.to_csv(
    pair_path,
    index=False,
)

print("\n" + "=" * 90)
print("FILES SAVED")
print("=" * 90)
print(single_path)
print(pair_path)
