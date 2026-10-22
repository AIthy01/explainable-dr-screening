import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

MANIFEST = "results/analysis/gradcam_case_manifest.csv"
BASE_DIR = "results/figures/gradcam_representative_cases"
OUTPUT = os.path.join(BASE_DIR, "gradcam_montage.png")

TARGET_WIDTH = 1100
TITLE_HEIGHT = 70
COLUMNS = 2

df = pd.read_csv(MANIFEST)

font = ImageFont.load_default()

cards = []

for _, row in df.iterrows():

    image_id = str(row["id_code"])

    path = os.path.join(
        BASE_DIR,
        image_id,
        "gradcam_comparison.png",
    )

    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    img = Image.open(path).convert("RGB")

    scale = TARGET_WIDTH / img.width
    new_height = int(img.height * scale)

    img = img.resize(
        (TARGET_WIDTH, new_height),
        Image.Resampling.LANCZOS,
    )

    card = Image.new(
        "RGB",
        (TARGET_WIDTH, TITLE_HEIGHT + new_height),
        "white",
    )

    draw = ImageDraw.Draw(card)

    title1 = (
        f"{row['selection_type']} | "
        f"ID={image_id}"
    )

    title2 = (
        f"true={int(row['true_grade'])} | "
        f"VGG={int(row['vgg_pred'])} | "
        f"ResNet={int(row['resnet_pred'])} | "
        f"ensemble={int(row['ensemble_pred'])} | "
        f"ref_score={float(row['referable_score']):.4f}"
    )

    draw.text(
        (10, 10),
        title1,
        fill="black",
        font=font,
    )

    draw.text(
        (10, 35),
        title2,
        fill="black",
        font=font,
    )

    card.paste(
        img,
        (0, TITLE_HEIGHT),
    )

    cards.append(card)


rows = (len(cards) + COLUMNS - 1) // COLUMNS

row_heights = []

for r in range(rows):

    group = cards[
        r * COLUMNS:
        (r + 1) * COLUMNS
    ]

    row_heights.append(
        max(card.height for card in group)
    )


canvas_width = TARGET_WIDTH * COLUMNS
canvas_height = sum(row_heights)

canvas = Image.new(
    "RGB",
    (canvas_width, canvas_height),
    "white",
)

y = 0

for r in range(rows):

    group = cards[
        r * COLUMNS:
        (r + 1) * COLUMNS
    ]

    for c, card in enumerate(group):

        canvas.paste(
            card,
            (c * TARGET_WIDTH, y),
        )

    y += row_heights[r]


canvas.save(
    OUTPUT,
    quality=95,
)

print(f"Saved: {OUTPUT}")
