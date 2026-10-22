import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from config import TRAIN_CSV, IMAGE_DIR


# ============================================================
# Configuration
# ============================================================

PRED_DIR = "results/predictions"
OUT_DIR = "results/analysis"

W_VGG = 0.60
W_RESNET = 0.40
REFERRAL_THRESHOLD = 0.40

os.makedirs(OUT_DIR, exist_ok=True)


# ============================================================
# Reconstruct validation set in exactly the same order
# ============================================================

df = pd.read_csv(TRAIN_CSV)

_, val_idx = train_test_split(
    np.arange(len(df)),
    test_size=0.2,
    random_state=42,
    stratify=df["diagnosis"],
)

val_df = (
    df.iloc[val_idx]
    .copy()
    .reset_index(drop=True)
)

labels = np.load(
    os.path.join(PRED_DIR, "labels.npy")
)

vgg_probs = np.load(
    os.path.join(PRED_DIR, "vgg16_probs.npy")
)

resnet_probs = np.load(
    os.path.join(PRED_DIR, "resnet50_probs.npy")
)

assert len(val_df) == len(labels) == 586

assert np.array_equal(
    val_df["diagnosis"].to_numpy(),
    labels,
), "Validation order does not match saved predictions."


# ============================================================
# Final ensemble
# ============================================================

fused_probs = (
    W_VGG * vgg_probs
    + W_RESNET * resnet_probs
)

vgg_pred = np.argmax(
    vgg_probs,
    axis=1,
)

resnet_pred = np.argmax(
    resnet_probs,
    axis=1,
)

ensemble_pred = np.argmax(
    fused_probs,
    axis=1,
)

ensemble_top_score = np.max(
    fused_probs,
    axis=1,
)

referable_score = fused_probs[
    :,
    2:
].sum(axis=1)

true_referable = labels >= 2

pred_referable = (
    referable_score
    >= REFERRAL_THRESHOLD
)

# 两模型概率分布差异
prob_disagreement = np.abs(
    vgg_probs - resnet_probs
).sum(axis=1)


# ============================================================
# Build analysis table
# ============================================================

data = val_df[
    ["id_code", "diagnosis"]
].copy()

data["val_position"] = np.arange(
    len(data)
)

data["true_grade"] = labels

data["vgg_pred"] = vgg_pred
data["resnet_pred"] = resnet_pred
data["ensemble_pred"] = ensemble_pred

data["ensemble_top_score"] = (
    ensemble_top_score
)

data["referable_score"] = (
    referable_score
)

data["true_referable"] = (
    true_referable
)

data["pred_referable"] = (
    pred_referable
)

data["prob_disagreement_L1"] = (
    prob_disagreement
)

data["ensemble_correct"] = (
    ensemble_pred == labels
)

data["branches_agree"] = (
    vgg_pred == resnet_pred
)


# ============================================================
# Resolve image path
# ============================================================

def resolve_image_path(image_id):

    for ext in (
        ".png",
        ".jpg",
        ".jpeg",
        ".PNG",
        ".JPG",
        ".JPEG",
    ):
        path = os.path.join(
            IMAGE_DIR,
            str(image_id) + ext,
        )

        if os.path.isfile(path):
            return path

    raise FileNotFoundError(
        f"Image not found: {image_id}"
    )


data["image_path"] = data[
    "id_code"
].apply(resolve_image_path)


# ============================================================
# Selection helpers
# ============================================================

selected = []
used_positions = set()


def add_case(
    row,
    selection_type,
    reason,
):

    pos = int(
        row["val_position"]
    )

    if pos in used_positions:
        return False

    used_positions.add(pos)

    selected.append({
        "selection_type":
            selection_type,

        "selection_reason":
            reason,

        "val_position":
            pos,

        "id_code":
            row["id_code"],

        "image_path":
            row["image_path"],

        "true_grade":
            int(row["true_grade"]),

        "vgg_pred":
            int(row["vgg_pred"]),

        "resnet_pred":
            int(row["resnet_pred"]),

        "ensemble_pred":
            int(row["ensemble_pred"]),

        "ensemble_top_score":
            float(
                row["ensemble_top_score"]
            ),

        "referable_score":
            float(
                row["referable_score"]
            ),

        "true_referable":
            bool(
                row["true_referable"]
            ),

        "pred_referable":
            bool(
                row["pred_referable"]
            ),

        "prob_disagreement_L1":
            float(
                row["prob_disagreement_L1"]
            ),
    })

    return True


def choose_median_confidence(
    candidates,
):

    candidates = candidates[
        ~candidates["val_position"].isin(
            used_positions
        )
    ].copy()

    if len(candidates) == 0:
        return None

    median_score = candidates[
        "ensemble_top_score"
    ].median()

    idx = (
        candidates[
            "ensemble_top_score"
        ]
        .sub(median_score)
        .abs()
        .idxmin()
    )

    return candidates.loc[idx]


# ============================================================
# 1. One representative correctly classified case per grade
#
# 不选最高置信度，而选该等级正确病例中
# 接近中位置信度的样本，减少“挑最好看的案例”偏倚。
# ============================================================

for grade in range(5):

    candidates = data[
        (data["true_grade"] == grade)
        &
        (data["ensemble_correct"])
    ]

    row = choose_median_confidence(
        candidates
    )

    if row is not None:
        add_case(
            row,
            f"correct_grade_{grade}",
            (
                f"Correct grade {grade}; "
                "ensemble confidence closest "
                "to median among correctly "
                "classified cases of this grade."
            ),
        )


# ============================================================
# 2. Branches agree + ensemble correct
# ============================================================

candidates = data[
    data["branches_agree"]
    &
    data["ensemble_correct"]
    &
    ~data["val_position"].isin(
        used_positions
    )
]

row = choose_median_confidence(
    candidates
)

if row is not None:
    add_case(
        row,
        "branches_agree_correct",
        (
            "VGG16 and ResNet50 agree; "
            "ensemble prediction is correct; "
            "median-confidence representative."
        ),
    )


# ============================================================
# 3. Branches disagree, ensemble correct
#
# 选两个分支概率分布差异最大的一个，
# 用于观察融合是否利用了模型互补性。
# ============================================================

candidates = data[
    (~data["branches_agree"])
    &
    data["ensemble_correct"]
    &
    ~data["val_position"].isin(
        used_positions
    )
]

if len(candidates) > 0:

    row = candidates.loc[
        candidates[
            "prob_disagreement_L1"
        ].idxmax()
    ]

    add_case(
        row,
        "branches_disagree_ensemble_correct",
        (
            "VGG16 and ResNet50 disagree; "
            "ensemble is correct; selected "
            "for largest branch probability "
            "distribution disagreement."
        ),
    )


# ============================================================
# 4. Ensemble high-confidence classification error
#
# 故意选最高置信度错误病例作为 stress-test，
# 不代表错误病例的平均情况。
# ============================================================

candidates = data[
    (~data["ensemble_correct"])
    &
    ~data["val_position"].isin(
        used_positions
    )
]

if len(candidates) > 0:

    row = candidates.loc[
        candidates[
            "ensemble_top_score"
        ].idxmax()
    ]

    add_case(
        row,
        "ensemble_high_confidence_error",
        (
            "Incorrect ensemble prediction "
            "with highest ensemble top-class "
            "score; selected as failure-mode "
            "stress test."
        ),
    )


# ============================================================
# 5. Referable DR false positive
# ============================================================

candidates = data[
    (~data["true_referable"])
    &
    data["pred_referable"]
    &
    ~data["val_position"].isin(
        used_positions
    )
]

if len(candidates) > 0:

    row = candidates.loc[
        candidates[
            "referable_score"
        ].idxmax()
    ]

    add_case(
        row,
        "referable_false_positive",
        (
            "Referable-DR false positive "
            "with highest referable score."
        ),
    )


# ============================================================
# 6. Referable DR false negative
# ============================================================

candidates = data[
    data["true_referable"]
    &
    (~data["pred_referable"])
    &
    ~data["val_position"].isin(
        used_positions
    )
]

if len(candidates) > 0:

    row = candidates.loc[
        candidates[
            "referable_score"
        ].idxmin()
    ]

    add_case(
        row,
        "referable_false_negative",
        (
            "Referable-DR false negative "
            "with lowest referable score; "
            "selected as missed-case "
            "stress test."
        ),
    )


# ============================================================
# Save manifest
# ============================================================

manifest = pd.DataFrame(
    selected
)

manifest_path = os.path.join(
    OUT_DIR,
    "gradcam_case_manifest.csv",
)

manifest.to_csv(
    manifest_path,
    index=False,
)


# ============================================================
# Print
# ============================================================

pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    240,
)

print("=" * 100)
print("GRAD-CAM REPRESENTATIVE CASE MANIFEST")
print("=" * 100)
print(f"Selected cases: {len(manifest)}")
print()

display_cols = [
    "selection_type",
    "id_code",
    "true_grade",
    "vgg_pred",
    "resnet_pred",
    "ensemble_pred",
    "ensemble_top_score",
    "referable_score",
    "true_referable",
    "pred_referable",
    "prob_disagreement_L1",
]

print(
    manifest[
        display_cols
    ]
    .round(4)
    .to_string(index=False)
)

print()
print("=" * 100)
print("SAVED")
print("=" * 100)
print(manifest_path)
