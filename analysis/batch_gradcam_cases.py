import os
import pandas as pd

from predictor import DRPredictor
from ensemble_gradcam import explain_image


MANIFEST_PATH = (
    "results/analysis/"
    "gradcam_case_manifest.csv"
)

SAVE_DIR = (
    "results/figures/"
    "gradcam_representative_cases"
)


if not os.path.isfile(
    MANIFEST_PATH
):
    raise FileNotFoundError(
        MANIFEST_PATH
    )


manifest = pd.read_csv(
    MANIFEST_PATH
)

os.makedirs(
    SAVE_DIR,
    exist_ok=True,
)


# ============================================================
# Load VGG16 + ResNet50 only once
# ============================================================

predictor = DRPredictor()


print()
print("=" * 90)
print("BATCH GRAD-CAM")
print("=" * 90)
print(
    f"Cases: {len(manifest)}"
)
print(
    f"Output: {SAVE_DIR}"
)


result_rows = []


for i, row in manifest.iterrows():

    selection_type = str(
        row["selection_type"]
    )

    image_path = str(
        row["image_path"]
    )

    target_class = int(
        row["ensemble_pred"]
    )

    print()
    print(
        f"[{i + 1}/{len(manifest)}] "
        f"{selection_type}"
    )

    print(
        f"Image: {image_path}"
    )

    print(
        f"True grade: "
        f"{int(row['true_grade'])}"
    )

    print(
        f"Ensemble grade: "
        f"{target_class}"
    )

    result = explain_image(
        image_path=image_path,
        save_dir=SAVE_DIR,
        target_class=target_class,
        predictor=predictor,
    )

    result_rows.append({
        "selection_type":
            selection_type,

        "id_code":
            row["id_code"],

        "true_grade":
            int(row["true_grade"]),

        "ensemble_pred":
            target_class,

        "comparison_path":
            result[
                "comparison_path"
            ],

        "weighted_cam_path":
            result[
                "weighted_cam_path"
            ],
    })


result_df = pd.DataFrame(
    result_rows
)

result_path = os.path.join(
    SAVE_DIR,
    "batch_gradcam_results.csv",
)

result_df.to_csv(
    result_path,
    index=False,
)


print()
print("=" * 90)
print("BATCH GRAD-CAM FINISHED")
print("=" * 90)
print(result_path)
