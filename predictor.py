import os
import json
import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from model import get_model
from transforms import get_val_transform


# ============================================================
# Final development-stage system configuration
# ============================================================

CLASS_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

MODEL_WEIGHTS = {
    "vgg16": 0.60,
    "resnet50": 0.40,
}

CHECKPOINT_PATHS = {
    "vgg16": "results/checkpoints/vgg16_best.pth",
    "resnet50": "results/checkpoints/resnet50_best.pth",
}

# Development operating threshold
# Referable DR = grade 2 / 3 / 4
REFERRAL_THRESHOLD = 0.40


class DRPredictor:
    """
    Diabetic Retinopathy inference wrapper.

    Final development-stage ensemble:
        fused_probs =
            0.60 * VGG16_probs
            + 0.40 * ResNet50_probs

    Outputs:
        1. Five-class DR severity classification
        2. Referable DR score = P2 + P3 + P4
        3. Referral prompt based on development threshold t = 0.40

    Important:
        referable_dr_score is a model screening score,
        NOT a calibrated clinical probability.
    """

    def __init__(
        self,
        device=None,
        referral_threshold=REFERRAL_THRESHOLD,
    ):
        # ----------------------------------------------------
        # Device
        # ----------------------------------------------------

        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.referral_threshold = float(
            referral_threshold
        )

        # Use exactly the same validation transform
        self.transform = get_val_transform()

        # ----------------------------------------------------
        # Load models
        # ----------------------------------------------------

        self.models = {}

        for model_name, weight in MODEL_WEIGHTS.items():

            checkpoint_path = CHECKPOINT_PATHS[
                model_name
            ]

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(
                    f"Checkpoint not found: "
                    f"{checkpoint_path}"
                )

            model = get_model(
                model_name
            ).to(self.device)

            state_dict = torch.load(
                checkpoint_path,
                map_location=self.device,
            )

            model.load_state_dict(
                state_dict
            )

            model.eval()

            self.models[model_name] = {
                "model": model,
                "weight": float(weight),
            }

        print(
            "DRPredictor initialized successfully."
        )

        print(
            f"Device: {self.device}"
        )

        print(
            "Ensemble: "
            "VGG16=0.60, ResNet50=0.40"
        )

        print(
            "Referable DR threshold: "
            f"{self.referral_threshold:.2f}"
        )

    # ========================================================
    # Image loading
    # ========================================================

    def _load_image(self, image_input):

        if isinstance(
            image_input,
            (str, os.PathLike),
        ):

            image_path = str(
                image_input
            )

            if not os.path.exists(image_path):
                raise FileNotFoundError(
                    f"Image not found: {image_path}"
                )

            image = Image.open(
                image_path
            ).convert("RGB")

            return image

        if isinstance(
            image_input,
            Image.Image,
        ):
            return image_input.convert(
                "RGB"
            )

        raise TypeError(
            "image_input must be a file path "
            "or PIL.Image.Image."
        )

    # ========================================================
    # Single-model inference
    # ========================================================

    def _predict_single_model(
        self,
        model,
        image_tensor,
    ):

        with torch.inference_mode():

            logits = model(
                image_tensor
            )

            probs = F.softmax(
                logits,
                dim=1,
            )

        return probs[0]

    # ========================================================
    # Main prediction
    # ========================================================

    def predict(
        self,
        image_input,
    ):

        image = self._load_image(
            image_input
        )

        image_tensor = self.transform(
            image
        )

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(
                self.device,
                non_blocking=True,
            )
        )

        # ----------------------------------------------------
        # Individual model probabilities
        # ----------------------------------------------------

        individual_probs = {}

        fused_probs = torch.zeros(
            5,
            device=self.device,
        )

        for model_name, item in self.models.items():

            probs = self._predict_single_model(
                item["model"],
                image_tensor,
            )

            individual_probs[
                model_name
            ] = probs

            fused_probs += (
                item["weight"]
                * probs
            )

        # ----------------------------------------------------
        # Five-class DR severity classification
        # ----------------------------------------------------

        predicted_grade = int(
            torch.argmax(
                fused_probs
            ).item()
        )

        top_class_score = float(
            fused_probs[
                predicted_grade
            ].item()
        )

        # ----------------------------------------------------
        # Referable DR screening score
        #
        # grade 2 + grade 3 + grade 4
        # ----------------------------------------------------

        referable_dr_score = float(
            fused_probs[2:].sum().item()
        )

        referable_dr = bool(
            referable_dr_score
            >= self.referral_threshold
        )

        # ----------------------------------------------------
        # User-facing referral text
        # ----------------------------------------------------

        if referable_dr:

            referral_message = (
                "模型筛查结果达到可转诊级DR提示阈值，"
                "建议进一步接受专业眼科评估。"
            )

        else:

            referral_message = (
                "当前模型未触发可转诊级DR提示。"
                "该结果仅用于辅助筛查，"
                "不能替代专业眼科检查。"
            )

        # ----------------------------------------------------
        # Convert probabilities to ordinary Python floats
        # ----------------------------------------------------

        fused_probs_cpu = (
            fused_probs
            .detach()
            .cpu()
            .numpy()
        )

        class_probabilities = {
            f"grade_{i}_{CLASS_NAMES[i]}":
                float(
                    fused_probs_cpu[i]
                )
            for i in range(5)
        }

        model_probabilities = {}

        for model_name, probs in individual_probs.items():

            probs_cpu = (
                probs
                .detach()
                .cpu()
                .numpy()
            )

            model_probabilities[
                model_name
            ] = {
                f"grade_{i}_{CLASS_NAMES[i]}":
                    float(
                        probs_cpu[i]
                    )
                for i in range(5)
            }

        # ----------------------------------------------------
        # Structured output
        # ----------------------------------------------------

        result = {
            "predicted_grade":
                predicted_grade,

            "predicted_label":
                CLASS_NAMES[
                    predicted_grade
                ],

            # This is the maximum fused softmax score.
            # It is not claimed to be calibrated probability.
            "top_class_score":
                top_class_score,

            "class_probabilities":
                class_probabilities,

            "referable_dr_score":
                referable_dr_score,

            "referral_threshold":
                self.referral_threshold,

            "referable_dr":
                referable_dr,

            "referral_message":
                referral_message,

            "ensemble": {
                "vgg16_weight":
                    MODEL_WEIGHTS[
                        "vgg16"
                    ],

                "resnet50_weight":
                    MODEL_WEIGHTS[
                        "resnet50"
                    ],
            },

            "individual_model_probabilities":
                model_probabilities,
        }

        return result


# ============================================================
# Command-line interface
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Diabetic Retinopathy "
            "ensemble predictor"
        )
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to a fundus image.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Optional device, e.g. "
            "'cuda' or 'cpu'."
        ),
    )

    args = parser.parse_args()

    predictor = DRPredictor(
        device=args.device
    )

    result = predictor.predict(
        args.image
    )

    print("\n" + "=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
