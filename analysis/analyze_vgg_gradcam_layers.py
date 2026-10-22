import os
import numpy as np
import pandas as pd
import torch

from PIL import Image
from scipy.ndimage import label, binary_fill_holes

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from predictor import DRPredictor


MANIFEST = "results/analysis/gradcam_case_manifest.csv"

OUT_DIR = "results/analysis/vgg_gradcam_layer_test"

os.makedirs(
    OUT_DIR,
    exist_ok=True,
)

CANDIDATE_LAYERS = {
    "conv5_1_features24": 24,
    "conv5_2_features26": 26,
    "conv5_3_features28": 28,
    "relu5_3_features29": 29,
}

EPS = 1e-8


def make_retina_mask(rgb):

    brightness = rgb.mean(axis=2)

    mask = brightness > 0.05

    labeled, n = label(mask)

    if n > 0:

        sizes = np.bincount(
            labeled.ravel()
        )

        sizes[0] = 0

        largest = sizes.argmax()

        mask = labeled == largest

    mask = binary_fill_holes(
        mask
    )

    return mask.astype(bool)


def normalize_heatmap(h):

    h = np.asarray(
        h,
        dtype=np.float32,
    )

    minimum = float(h.min())
    maximum = float(h.max())

    if maximum - minimum < EPS:
        return np.zeros_like(h)

    return (
        (h - minimum)
        /
        (maximum - minimum)
    )


def is_degenerate(h):

    return (
        float(h.max()) < EPS
        or
        float(h.sum()) < EPS
        or
        float(h.std()) < EPS
    )


def outside_mass(
    h,
    mask,
):

    total = float(
        h.sum()
    )

    if total < EPS:
        return np.nan

    return float(
        h[~mask].sum()
        /
        total
    )


def entropy(
    h,
    mask,
):

    values = h[mask].astype(
        np.float64
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

    return float(
        -(
            p * np.log(p)
        ).sum()
        /
        np.log(len(values))
    )


def area80(
    h,
    mask,
):

    values = h[mask].astype(
        np.float64
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

    k = (
        np.searchsorted(
            cumulative,
            0.80 * total,
        )
        + 1
    )

    return float(
        k / len(values)
    )


manifest = pd.read_csv(
    MANIFEST
)

predictor = DRPredictor()

model = predictor.models[
    "vgg16"
]["model"]

rows = []


for _, row in manifest.iterrows():

    image_path = str(
        row["image_path"]
    )

    image_id = str(
        row["id_code"]
    )

    target_class = int(
        row["ensemble_pred"]
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    display = image.resize(
        (224, 224)
    )

    rgb = (
        np.asarray(display)
        .astype(np.float32)
        / 255.0
    )

    retina_mask = make_retina_mask(
        rgb
    )

    tensor = (
        predictor.transform(image)
        .unsqueeze(0)
        .to(predictor.device)
    )

    for layer_name, layer_index in CANDIDATE_LAYERS.items():

        target_layer = model.features[
            layer_index
        ]

        cam = GradCAM(
            model=model,
            target_layers=[
                target_layer
            ],
        )

        grayscale = cam(
            input_tensor=tensor,
            targets=[
                ClassifierOutputTarget(
                    target_class
                )
            ],
        )[0]

        h = normalize_heatmap(
            grayscale
        )

        degenerate = is_degenerate(
            h
        )

        rows.append({
            "selection_type":
                row["selection_type"],

            "id_code":
                image_id,

            "target_class":
                target_class,

            "layer_name":
                layer_name,

            "layer_index":
                layer_index,

            "degenerate":
                degenerate,

            "max":
                float(h.max()),

            "mean":
                float(h.mean()),

            "std":
                float(h.std()),

            "outside_mass":
                outside_mass(
                    h,
                    retina_mask,
                ),

            "entropy":
                entropy(
                    h,
                    retina_mask,
                ),

            "area80":
                area80(
                    h,
                    retina_mask,
                ),
        })


result = pd.DataFrame(
    rows
)

raw_path = os.path.join(
    OUT_DIR,
    "vgg_layer_qc_raw.csv",
)

result.to_csv(
    raw_path,
    index=False,
)


summary_rows = []

for layer_name in CANDIDATE_LAYERS:

    temp = result[
        result["layer_name"]
        == layer_name
    ]

    valid = temp[
        ~temp["degenerate"]
    ]

    summary_rows.append({
        "layer_name":
            layer_name,

        "degenerate_count":
            int(
                temp[
                    "degenerate"
                ].sum()
            ),

        "valid_count":
            int(
                len(valid)
            ),

        "mean_outside_mass":
            valid[
                "outside_mass"
            ].mean(),

        "mean_entropy":
            valid[
                "entropy"
            ].mean(),

        "mean_area80":
            valid[
                "area80"
            ].mean(),
    })


summary = pd.DataFrame(
    summary_rows
)

summary_path = os.path.join(
    OUT_DIR,
    "vgg_layer_qc_summary.csv",
)

summary.to_csv(
    summary_path,
    index=False,
)


pd.set_option(
    "display.max_columns",
    None,
)

pd.set_option(
    "display.width",
    200,
)

print("=" * 100)
print("VGG16 GRAD-CAM TARGET-LAYER TEST")
print("=" * 100)

print(
    summary
    .round(4)
    .to_string(index=False)
)

print()
print("Saved:")
print(raw_path)
print(summary_path)
