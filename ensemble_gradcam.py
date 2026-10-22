import os
import json
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from predictor import (
    DRPredictor,
    CLASS_NAMES,
)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 224

W_VGG = 0.60
W_RESNET = 0.40

DEFAULT_SAVE_DIR = "results/figures/gradcam"


# ============================================================
# Target layers
# ============================================================

def get_target_layer(model_name, model):
    """
    Choose the last meaningful convolutional block.

    VGG16:
        features[28] = last Conv2d (conv5_3)
        features[-1] would be MaxPool, so do NOT use it.

    ResNet50:
        layer4[-1] = last Bottleneck block.
    """

    if model_name == "vgg16":
        # conv5_2:
        # target-layer sensitivity analysis showed
        # 0/10 degenerate CAMs and the lowest
        # mean outside-retina attention among
        # stable candidate layers.
        return model.features[26]

    if model_name == "resnet50":
        return model.layer4[-1]

    raise ValueError(
        f"Unsupported model for current ensemble: {model_name}"
    )


# ============================================================
# Image preparation
# ============================================================

def prepare_image(image_path, transform, device):

    if not os.path.isfile(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = Image.open(
        image_path
    ).convert("RGB")

    # Exactly 224x224 for visualization,
    # consistent with validation preprocessing.
    display_image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    rgb_image = (
        np.asarray(
            display_image
        )
        .astype(np.float32)
        / 255.0
    )

    input_tensor = (
        transform(image)
        .unsqueeze(0)
        .to(device)
    )

    return image, rgb_image, input_tensor


# ============================================================
# Grad-CAM
# ============================================================

def generate_gradcam(
    model,
    target_layer,
    input_tensor,
    rgb_image,
    target_class,
):
    """
    Explain a SPECIFIED class.

    Both VGG16 and ResNet50 are therefore forced to explain
    the same ensemble-selected DR grade.
    """

    targets = [
        ClassifierOutputTarget(
            target_class
        )
    ]

    cam = GradCAM(
        model=model,
        target_layers=[
            target_layer
        ],
    )

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets,
    )[0]

    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True,
    )

    return (
        grayscale_cam.astype(
            np.float32
        ),
        visualization,
    )


def normalize_heatmap(heatmap):

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32,
    )

    min_value = float(
        heatmap.min()
    )

    max_value = float(
        heatmap.max()
    )

    if max_value - min_value < 1e-12:
        return np.zeros_like(
            heatmap
        )

    return (
        (heatmap - min_value)
        / (max_value - min_value)
    )


# ============================================================
# Main explanation function
# ============================================================

def explain_image(
    image_path,
    save_dir=DEFAULT_SAVE_DIR,
    target_class=None,
    device=None,
    predictor=None,
):

    # --------------------------------------------------------
    # Load exactly the frozen final inference system.
    # A preloaded predictor may be supplied for batch use.
    # --------------------------------------------------------

    if predictor is None:
        predictor = DRPredictor(
            device=device
        )

    prediction_result = predictor.predict(
        image_path
    )

    ensemble_predicted_class = int(
        prediction_result[
            "predicted_grade"
        ]
    )

    # By default explain the final ensemble-predicted class.
    if target_class is None:
        target_class = (
            ensemble_predicted_class
        )

    target_class = int(
        target_class
    )

    if target_class not in range(5):
        raise ValueError(
            "target_class must be one of 0,1,2,3,4."
        )

    # --------------------------------------------------------
    # Prepare image
    # --------------------------------------------------------

    _, rgb_image, input_tensor = prepare_image(
        image_path,
        predictor.transform,
        predictor.device,
    )

    vgg_model = predictor.models[
        "vgg16"
    ]["model"]

    resnet_model = predictor.models[
        "resnet50"
    ]["model"]

    # --------------------------------------------------------
    # Grad-CAM for SAME target class
    # --------------------------------------------------------

    vgg_heat, vgg_cam = generate_gradcam(
        model=vgg_model,
        target_layer=get_target_layer(
            "vgg16",
            vgg_model,
        ),
        input_tensor=input_tensor,
        rgb_image=rgb_image,
        target_class=target_class,
    )

    resnet_heat, resnet_cam = generate_gradcam(
        model=resnet_model,
        target_layer=get_target_layer(
            "resnet50",
            resnet_model,
        ),
        input_tensor=input_tensor,
        rgb_image=rgb_image,
        target_class=target_class,
    )

    # --------------------------------------------------------
    # Weighted CAM summary
    #
    # IMPORTANT:
    # This is NOT claimed to be a mathematically unique
    # "ensemble Grad-CAM".
    #
    # It is a visualization summary of two branch-specific
    # Grad-CAM maps using the frozen ensemble weights.
    # --------------------------------------------------------

    vgg_heat_norm = normalize_heatmap(
        vgg_heat
    )

    resnet_heat_norm = normalize_heatmap(
        resnet_heat
    )

    weighted_heat = (
        W_VGG * vgg_heat_norm
        + W_RESNET * resnet_heat_norm
    )

    weighted_heat = normalize_heatmap(
        weighted_heat
    )

    weighted_cam = show_cam_on_image(
        rgb_image,
        weighted_heat,
        use_rgb=True,
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    image_stem = os.path.splitext(
        os.path.basename(
            image_path
        )
    )[0]

    image_save_dir = os.path.join(
        save_dir,
        image_stem,
    )

    os.makedirs(
        image_save_dir,
        exist_ok=True,
    )

    # Save raw normalized CAM arrays for quantitative QC
    np.save(
        os.path.join(image_save_dir, "vgg16_heatmap.npy"),
        vgg_heat_norm,
    )

    np.save(
        os.path.join(image_save_dir, "resnet50_heatmap.npy"),
        resnet_heat_norm,
    )

    np.save(
        os.path.join(image_save_dir, "weighted_cam_heatmap.npy"),
        weighted_heat,
    )

    # --------------------------------------------------------
    # Save original + heatmaps + overlays
    # --------------------------------------------------------

    plt.imsave(
        os.path.join(
            image_save_dir,
            "original.png",
        ),
        rgb_image,
    )

    plt.imsave(
        os.path.join(
            image_save_dir,
            "vgg16_heatmap.png",
        ),
        vgg_heat_norm,
        cmap="jet",
    )

    plt.imsave(
        os.path.join(
            image_save_dir,
            "vgg16_gradcam.png",
        ),
        vgg_cam,
    )

    plt.imsave(
        os.path.join(
            image_save_dir,
            "resnet50_heatmap.png",
        ),
        resnet_heat_norm,
        cmap="jet",
    )

    plt.imsave(
        os.path.join(
            image_save_dir,
            "resnet50_gradcam.png",
        ),
        resnet_cam,
    )

    plt.imsave(
        os.path.join(
            image_save_dir,
            "weighted_cam_heatmap.png",
        ),
        weighted_heat,
        cmap="jet",
    )

    plt.imsave(
        os.path.join(
            image_save_dir,
            "weighted_cam_summary.png",
        ),
        weighted_cam,
    )

    # --------------------------------------------------------
    # Create a comparison panel
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(16, 4),
    )

    axes[0].imshow(
        rgb_image
    )
    axes[0].set_title(
        "Original"
    )

    axes[1].imshow(
        vgg_cam
    )
    axes[1].set_title(
        "VGG16 Grad-CAM"
    )

    axes[2].imshow(
        resnet_cam
    )
    axes[2].set_title(
        "ResNet50 Grad-CAM"
    )

    axes[3].imshow(
        weighted_cam
    )
    axes[3].set_title(
        "Weighted CAM Summary"
    )

    for ax in axes:
        ax.axis("off")

    fig.suptitle(
        (
            f"Target grade: {target_class} "
            f"({CLASS_NAMES[target_class]})"
        ),
        fontsize=13,
    )

    fig.tight_layout()

    comparison_path = os.path.join(
        image_save_dir,
        "gradcam_comparison.png",
    )

    fig.savefig(
        comparison_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "image_path":
            image_path,

        "ensemble_predicted_grade":
            ensemble_predicted_class,

        "ensemble_predicted_label":
            CLASS_NAMES[
                ensemble_predicted_class
            ],

        "gradcam_target_grade":
            target_class,

        "gradcam_target_label":
            CLASS_NAMES[
                target_class
            ],

        "referable_dr_score":
            prediction_result[
                "referable_dr_score"
            ],

        "referral_threshold":
            prediction_result[
                "referral_threshold"
            ],

        "referable_dr":
            prediction_result[
                "referable_dr"
            ],

        "ensemble_weights": {
            "vgg16": W_VGG,
            "resnet50": W_RESNET,
        },

        "note": (
            "weighted_cam_summary is a weighted visualization "
            "summary of branch-specific Grad-CAM maps, "
            "not a unique mathematical ensemble Grad-CAM."
        ),
    }

    metadata_path = os.path.join(
        image_save_dir,
        "metadata.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2,
        )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()
    print("=" * 80)
    print("GRAD-CAM EXPLANATION")
    print("=" * 80)

    print(
        f"Image                  : {image_path}"
    )

    print(
        f"Ensemble prediction    : "
        f"{ensemble_predicted_class} "
        f"({CLASS_NAMES[ensemble_predicted_class]})"
    )

    print(
        f"Grad-CAM target class  : "
        f"{target_class} "
        f"({CLASS_NAMES[target_class]})"
    )

    print(
        f"Referable DR score     : "
        f"{prediction_result['referable_dr_score']:.4f}"
    )

    print(
        f"Referral threshold     : "
        f"{prediction_result['referral_threshold']:.2f}"
    )

    print(
        f"Referral triggered     : "
        f"{prediction_result['referable_dr']}"
    )

    print()
    print(
        f"Saved to: {image_save_dir}"
    )

    print()
    print(
        "Generated:"
    )

    for filename in [
        "original.png",
        "vgg16_heatmap.png",
        "vgg16_gradcam.png",
        "resnet50_heatmap.png",
        "resnet50_gradcam.png",
        "weighted_cam_heatmap.png",
        "weighted_cam_summary.png",
        "gradcam_comparison.png",
        "metadata.json",
    ]:
        print(
            f"  - {filename}"
        )

    return {
        "prediction":
            prediction_result,

        "target_class":
            target_class,

        "save_dir":
            image_save_dir,

        "comparison_path":
            comparison_path,

        "weighted_cam_path":
            os.path.join(
                image_save_dir,
                "weighted_cam_summary.png",
            ),
    }


# ============================================================
# CLI
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Grad-CAM explanation for the frozen "
            "VGG16 + ResNet50 DR ensemble."
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to fundus image.",
    )

    parser.add_argument(
        "--target-class",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=None,
        help=(
            "Class to explain. "
            "Default: final ensemble-predicted class."
        ),
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    explain_image(
        image_path=args.image,
        save_dir=args.save_dir,
        target_class=args.target_class,
        device=args.device,
    )


if __name__ == "__main__":
    main()
