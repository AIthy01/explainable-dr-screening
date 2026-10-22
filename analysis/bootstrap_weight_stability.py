import os
import itertools
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)

# ============================================================
# Configuration
# ============================================================

PRED_DIR = "results/predictions"
OUT_DIR = "results/bootstrap"

CANDIDATE_WEIGHTS = [
    0.50,
    0.55,
    0.60,
    0.65,
]

N_BOOTSTRAP = 1000
SEED = 42

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Load data
# ============================================================

labels = np.load(
    os.path.join(
        PRED_DIR,
        "labels.npy",
    )
)

vgg_probs = np.load(
    os.path.join(
        PRED_DIR,
        "vgg16_probs.npy",
    )
)

resnet_probs = np.load(
    os.path.join(
        PRED_DIR,
        "resnet50_probs.npy",
    )
)

if vgg_probs.shape != resnet_probs.shape:
    raise ValueError(
        "VGG16 and ResNet50 probability shapes differ."
    )

if len(labels) != len(vgg_probs):
    raise ValueError(
        "Number of labels does not match probabilities."
    )

print("=" * 90)
print("PAIRED STRATIFIED BOOTSTRAP")
print("=" * 90)
print(f"Samples        : {len(labels)}")
print(f"Bootstrap runs : {N_BOOTSTRAP}")
print(f"Candidate VGG weights: {CANDIDATE_WEIGHTS}")
print()


# ============================================================
# Precompute predictions / probabilities for every weight
# ============================================================

candidate_outputs = {}

for w_vgg in CANDIDATE_WEIGHTS:

    w_resnet = 1.0 - w_vgg

    fused_probs = (
        w_vgg * vgg_probs
        + w_resnet * resnet_probs
    )

    fused_preds = np.argmax(
        fused_probs,
        axis=1,
    )

    # 独立筛查概率
    referable_probs = fused_probs[:, 2:].sum(axis=1)
    severe_probs = fused_probs[:, 3:].sum(axis=1)

    candidate_outputs[w_vgg] = {
        "probs": fused_probs,
        "preds": fused_preds,
        "referable_probs": referable_probs,
        "severe_probs": severe_probs,
    }


# ============================================================
# Metric calculation
# ============================================================

def calculate_metrics(
    y_true,
    preds,
    fused_probs,
    referable_probs,
    severe_probs,
):

    # 五分类
    accuracy = accuracy_score(
        y_true,
        preds,
    )

    macro_f1 = f1_score(
        y_true,
        preds,
        average="macro",
        zero_division=0,
    )

    multiclass_auc = roc_auc_score(
        y_true,
        fused_probs,
        multi_class="ovr",
        average="macro",
    )

    # --------------------------------------------------------
    # Referable DR: >= 2
    # --------------------------------------------------------

    y_ref = (
        y_true >= 2
    ).astype(int)

    pred_ref = (
        preds >= 2
    ).astype(int)

    referable_recall = recall_score(
        y_ref,
        pred_ref,
        zero_division=0,
    )

    referable_f1 = f1_score(
        y_ref,
        pred_ref,
        zero_division=0,
    )

    referable_auc = roc_auc_score(
        y_ref,
        referable_probs,
    )

    # --------------------------------------------------------
    # Severe-or-worse DR: >= 3
    # --------------------------------------------------------

    y_severe = (
        y_true >= 3
    ).astype(int)

    pred_severe = (
        preds >= 3
    ).astype(int)

    severe_recall = recall_score(
        y_severe,
        pred_severe,
        zero_division=0,
    )

    severe_f1 = f1_score(
        y_severe,
        pred_severe,
        zero_division=0,
    )

    severe_auc = roc_auc_score(
        y_severe,
        severe_probs,
    )

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "multiclass_auc": multiclass_auc,

        "referable_recall": referable_recall,
        "referable_f1": referable_f1,
        "referable_auc": referable_auc,

        "severe_recall": severe_recall,
        "severe_f1": severe_f1,
        "severe_auc": severe_auc,
    }


# ============================================================
# Original-sample metrics
# ============================================================

original_rows = []

for w_vgg in CANDIDATE_WEIGHTS:

    out = candidate_outputs[w_vgg]

    metrics = calculate_metrics(
        labels,
        out["preds"],
        out["probs"],
        out["referable_probs"],
        out["severe_probs"],
    )

    original_rows.append(
        {
            "weight_vgg": w_vgg,
            "weight_resnet": 1.0 - w_vgg,
            **metrics,
        }
    )

original_df = pd.DataFrame(
    original_rows
)


# ============================================================
# Paired stratified bootstrap
#
# 对每一个真实类别分别有放回采样，
# 保持原始类别数量不变。
#
# 同一个 bootstrap 样本同时用于全部候选权重，
# 因此比较属于 paired comparison。
# ============================================================

rng = np.random.default_rng(SEED)

class_indices = {
    cls: np.where(labels == cls)[0]
    for cls in np.unique(labels)
}

bootstrap_rows = []

for b in range(N_BOOTSTRAP):

    sampled_parts = []

    for cls, indices in class_indices.items():

        sampled = rng.choice(
            indices,
            size=len(indices),
            replace=True,
        )

        sampled_parts.append(
            sampled
        )

    bootstrap_idx = np.concatenate(
        sampled_parts
    )

    y_boot = labels[
        bootstrap_idx
    ]

    for w_vgg in CANDIDATE_WEIGHTS:

        out = candidate_outputs[
            w_vgg
        ]

        metrics = calculate_metrics(
            y_boot,
            out["preds"][bootstrap_idx],
            out["probs"][bootstrap_idx],
            out["referable_probs"][bootstrap_idx],
            out["severe_probs"][bootstrap_idx],
        )

        bootstrap_rows.append(
            {
                "bootstrap_id": b,
                "weight_vgg": w_vgg,
                "weight_resnet": 1.0 - w_vgg,
                **metrics,
            }
        )


bootstrap_df = pd.DataFrame(
    bootstrap_rows
)


# ============================================================
# Bootstrap summary / 95% CI
# ============================================================

METRICS = [
    "accuracy",
    "macro_f1",
    "multiclass_auc",
    "referable_recall",
    "referable_f1",
    "referable_auc",
    "severe_recall",
    "severe_f1",
    "severe_auc",
]

summary_rows = []

for w_vgg in CANDIDATE_WEIGHTS:

    temp = bootstrap_df[
        bootstrap_df["weight_vgg"]
        == w_vgg
    ]

    for metric in METRICS:

        values = temp[
            metric
        ].to_numpy()

        original_value = original_df.loc[
            original_df["weight_vgg"]
            == w_vgg,
            metric,
        ].iloc[0]

        summary_rows.append(
            {
                "weight_vgg": w_vgg,
                "weight_resnet":
                    1.0 - w_vgg,
                "metric": metric,
                "original": original_value,
                "bootstrap_mean":
                    np.mean(values),
                "ci_lower":
                    np.percentile(
                        values,
                        2.5,
                    ),
                "ci_upper":
                    np.percentile(
                        values,
                        97.5,
                    ),
            }
        )

summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# Paired differences
# ============================================================

pairwise_rows = []

for w1, w2 in itertools.combinations(
    CANDIDATE_WEIGHTS,
    2,
):

    df1 = (
        bootstrap_df[
            bootstrap_df["weight_vgg"]
            == w1
        ]
        .sort_values("bootstrap_id")
        .reset_index(drop=True)
    )

    df2 = (
        bootstrap_df[
            bootstrap_df["weight_vgg"]
            == w2
        ]
        .sort_values("bootstrap_id")
        .reset_index(drop=True)
    )

    for metric in METRICS:

        delta = (
            df1[metric].to_numpy()
            - df2[metric].to_numpy()
        )

        pairwise_rows.append(
            {
                "weight_1_vgg": w1,
                "weight_2_vgg": w2,
                "metric": metric,
                "mean_delta_w1_minus_w2":
                    np.mean(delta),
                "ci_lower":
                    np.percentile(
                        delta,
                        2.5,
                    ),
                "ci_upper":
                    np.percentile(
                        delta,
                        97.5,
                    ),
                "prob_w1_better":
                    np.mean(delta > 0),
                "prob_equal":
                    np.mean(delta == 0),
            }
        )

pairwise_df = pd.DataFrame(
    pairwise_rows
)


# ============================================================
# Win rate
# ============================================================

win_rows = []

for metric in METRICS:

    pivot = bootstrap_df.pivot(
        index="bootstrap_id",
        columns="weight_vgg",
        values=metric,
    )

    for w in CANDIDATE_WEIGHTS:

        # ties are counted as wins for every tied candidate
        max_values = pivot.max(
            axis=1
        )

        win_rate = np.mean(
            np.isclose(
                pivot[w].to_numpy(),
                max_values.to_numpy(),
            )
        )

        win_rows.append(
            {
                "metric": metric,
                "weight_vgg": w,
                "weight_resnet":
                    1.0 - w,
                "win_rate": win_rate,
            }
        )

win_df = pd.DataFrame(
    win_rows
)


# ============================================================
# Save
# ============================================================

original_df.to_csv(
    os.path.join(
        OUT_DIR,
        "candidate_original_metrics.csv",
    ),
    index=False,
)

bootstrap_df.to_csv(
    os.path.join(
        OUT_DIR,
        "bootstrap_raw.csv",
    ),
    index=False,
)

summary_df.to_csv(
    os.path.join(
        OUT_DIR,
        "bootstrap_summary.csv",
    ),
    index=False,
)

pairwise_df.to_csv(
    os.path.join(
        OUT_DIR,
        "bootstrap_pairwise_differences.csv",
    ),
    index=False,
)

win_df.to_csv(
    os.path.join(
        OUT_DIR,
        "bootstrap_win_rates.csv",
    ),
    index=False,
)


# ============================================================
# Print useful results
# ============================================================

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    220,
)

print("=" * 90)
print("ORIGINAL METRICS")
print("=" * 90)

print(
    original_df
    .round(4)
    .to_string(index=False)
)


print("\n")
print("=" * 90)
print("95% BOOTSTRAP CI: KEY METRICS")
print("=" * 90)

key_metrics = [
    "accuracy",
    "macro_f1",
    "referable_recall",
    "referable_f1",
    "referable_auc",
    "severe_f1",
    "severe_auc",
]

key_summary = summary_df[
    summary_df["metric"].isin(
        key_metrics
    )
]

for metric in key_metrics:

    print(
        f"\n--- {metric} ---"
    )

    temp = key_summary[
        key_summary["metric"]
        == metric
    ][
        [
            "weight_vgg",
            "weight_resnet",
            "original",
            "bootstrap_mean",
            "ci_lower",
            "ci_upper",
        ]
    ]

    print(
        temp
        .round(4)
        .to_string(index=False)
    )


print("\n")
print("=" * 90)
print("PAIRED COMPARISON: VGG 0.60 vs 0.65")
print("=" * 90)

focus = pairwise_df[
    (
        pairwise_df["weight_1_vgg"]
        == 0.60
    )
    &
    (
        pairwise_df["weight_2_vgg"]
        == 0.65
    )
][
    [
        "metric",
        "mean_delta_w1_minus_w2",
        "ci_lower",
        "ci_upper",
        "prob_w1_better",
    ]
]

print(
    focus
    .round(4)
    .to_string(index=False)
)


print("\n")
print("=" * 90)
print("BOOTSTRAP WIN RATES")
print("=" * 90)

for metric in key_metrics:

    print(
        f"\n--- {metric} ---"
    )

    temp = (
        win_df[
            win_df["metric"]
            == metric
        ]
        .sort_values(
            "weight_vgg"
        )
    )

    print(
        temp[
            [
                "weight_vgg",
                "weight_resnet",
                "win_rate",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )


print("\n")
print("=" * 90)
print("FILES SAVED")
print("=" * 90)

for filename in [
    "candidate_original_metrics.csv",
    "bootstrap_raw.csv",
    "bootstrap_summary.csv",
    "bootstrap_pairwise_differences.csv",
    "bootstrap_win_rates.csv",
]:
    print(
        os.path.join(
            OUT_DIR,
            filename,
        )
    )
