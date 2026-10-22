import os
import numpy as np
import pandas as pd
from PIL import Image

from scipy.ndimage import (
    label,
    binary_fill_holes,
)

MANIFEST = (
    "results/analysis/"
    "gradcam_case_manifest.csv"
)

BASE_DIR = (
    "results/figures/"
    "gradcam_representative_cases"
)

OUT = (
    "results/analysis/"
    "gradcam_qc_metrics_v2.csv"
)

EPS = 1e-8
TOP_FRACTION = 0.20
MASS_TARGET = 0.80


# ============================================================
# Retina mask
# ============================================================

def make_retina_mask(rgb):

    brightness = rgb.mean(axis=2)

    # Basic foreground
    mask = brightness > 0.05

    # Keep largest connected component
    labeled, n = label(mask)

    if n > 0:
        sizes = np.bincount(
            labeled.ravel()
        )

        sizes[0] = 0

        largest = sizes.argmax()

        mask = (
            labeled == largest
        )

    mask = binary_fill_holes(
        mask
    )

    return mask.astype(bool)


# ============================================================
# CAM metrics
# ============================================================

def is_degenerate(heat):

    return (
        float(heat.max()) < EPS
        or
        float(heat.sum()) < EPS
        or
        float(heat.std()) < EPS
    )


def outside_mass(
    heat,
    retina_mask,
):

    total = float(
        heat.sum()
    )

    if total < EPS:
        return np.nan

    return float(
        heat[
            ~retina_mask
        ].sum()
        /
        total
    )


def mass_area_ratio(
    heat,
    retina_mask,
    target_mass=0.80,
):

    values = (
        heat[
            retina_mask
        ]
        .astype(np.float64)
    )

    values = np.clip(
        values,
        0,
        None,
    )

    total = values.sum()

    if total < EPS:
        return np.nan

    values = np.sort(
        values
    )[::-1]

    cumulative = np.cumsum(
        values
    )

    k = np.searchsorted(
        cumulative,
        target_mass * total,
    ) + 1

    return float(
        k / len(values)
    )


def spatial_entropy(
    heat,
    retina_mask,
):

    values = (
        heat[
            retina_mask
        ]
        .astype(np.float64)
    )

    values = np.clip(
        values,
        0,
        None,
    )

    total = values.sum()

    if total < EPS:
        return np.nan

    p = values / total
    p = p[p > 0]

    h = -(
        p * np.log(p)
    ).sum()

    return float(
        h
        /
        np.log(len(values))
    )


def fixed_top_indices(
    heat,
    retina_mask,
    fraction=0.20,
):

    flat_indices = np.flatnonzero(
        retina_mask.ravel()
    )

    values = (
        heat.ravel()[
            flat_indices
        ]
    )

    k = max(
        1,
        int(
            round(
                fraction
                * len(flat_indices)
            )
        ),
    )

    # Exactly k pixels, rather than percentile threshold.
    order = np.argsort(
        values,
        kind="stable",
    )

    selected = flat_indices[
        order[-k:]
    ]

    mask = np.zeros(
        heat.size,
        dtype=bool,
    )

    mask[selected] = True

    return mask.reshape(
        heat.shape
    )


def top_iou(
    heat_a,
    heat_b,
    retina_mask,
):

    if (
        is_degenerate(heat_a)
        or is_degenerate(heat_b)
    ):
        return np.nan

    a = fixed_top_indices(
        heat_a,
        retina_mask,
        TOP_FRACTION,
    )

    b = fixed_top_indices(
        heat_b,
        retina_mask,
        TOP_FRACTION,
    )

    intersection = np.logical_and(
        a,
        b,
    ).sum()

    union = np.logical_or(
        a,
        b,
    ).sum()

    return float(
        intersection / union
    )


# ============================================================
# Main
# ============================================================

manifest = pd.read_csv(
    MANIFEST
)

rows = []

for _, row in manifest.iterrows():

    image_id = str(
        row["id_code"]
    )

    case_dir = os.path.join(
        BASE_DIR,
        image_id,
    )

    original = np.asarray(
        Image.open(
            os.path.join(
                case_dir,
                "original.png",
            )
        ).convert("RGB")
    ).astype(
        np.float32
    ) / 255.0

    retina_mask = make_retina_mask(
        original
    )

    vgg = np.load(
        os.path.join(
            case_dir,
            "vgg16_heatmap.npy",
        )
    )

    resnet = np.load(
        os.path.join(
            case_dir,
            "resnet50_heatmap.npy",
        )
    )

    vgg_deg = is_degenerate(
        vgg
    )

    resnet_deg = is_degenerate(
        resnet
    )

    rows.append({
        "selection_type":
            row["selection_type"],

        "id_code":
            image_id,

        "true_grade":
            int(row["true_grade"]),

        "ensemble_pred":
            int(row["ensemble_pred"]),

        "vgg_degenerate":
            vgg_deg,

        "resnet_degenerate":
            resnet_deg,

        "vgg_max":
            float(vgg.max()),

        "resnet_max":
            float(resnet.max()),

        "vgg_outside_mass":
            outside_mass(
                vgg,
                retina_mask,
            ),

        "resnet_outside_mass":
            outside_mass(
                resnet,
                retina_mask,
            ),

        "vgg_entropy":
            spatial_entropy(
                vgg,
                retina_mask,
            ),

        "resnet_entropy":
            spatial_entropy(
                resnet,
                retina_mask,
            ),

        "vgg_area80":
            mass_area_ratio(
                vgg,
                retina_mask,
                MASS_TARGET,
            ),

        "resnet_area80":
            mass_area_ratio(
                resnet,
                retina_mask,
                MASS_TARGET,
            ),

        "top20_iou":
            top_iou(
                vgg,
                resnet,
                retina_mask,
            ),
    })


result = pd.DataFrame(
    rows
)

result.to_csv(
    OUT,
    index=False,
)


pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    240,
)


print("=" * 110)
print("GRAD-CAM QC V2")
print("=" * 110)

print(
    result
    .round(4)
    .to_string(index=False)
)


print()
print("=" * 110)
print("MODEL SUMMARY")
print("=" * 110)

print(
    "VGG degenerate maps:",
    int(
        result[
            "vgg_degenerate"
        ].sum()
    ),
    "/",
    len(result),
)

print(
    "ResNet degenerate maps:",
    int(
        result[
            "resnet_degenerate"
        ].sum()
    ),
    "/",
    len(result),
)

for model in [
    "vgg",
    "resnet",
]:

    valid = result[
        ~result[
            f"{model}_degenerate"
        ]
    ]

    print()
    print(
        model.upper()
    )

    print(
        "mean outside mass:",
        round(
            valid[
                f"{model}_outside_mass"
            ].mean(),
            4,
        ),
    )

    print(
        "mean entropy:",
        round(
            valid[
                f"{model}_entropy"
            ].mean(),
            4,
        ),
    )

    print(
        "mean area80:",
        round(
            valid[
                f"{model}_area80"
            ].mean(),
            4,
        ),
    )


valid_iou = result[
    "top20_iou"
].dropna()

print()
print(
    "Valid pairwise CAM comparisons:",
    len(valid_iou),
)

print(
    "Mean fixed-top20 IoU:",
    round(
        valid_iou.mean(),
        4,
    )
)

print(
    "Median fixed-top20 IoU:",
    round(
        valid_iou.median(),
        4,
    )
)

print()
print("Saved:", OUT)
