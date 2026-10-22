import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class DRDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

        # 基本格式检查
        required_columns = {"id_code", "diagnosis"}
        missing_columns = required_columns - set(self.data.columns)

        if missing_columns:
            raise ValueError(
                f"CSV 缺少必要列: {sorted(missing_columns)}"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]

        # APTOS 图像 ID
        image_id = str(row["id_code"])

        # 自动兼容常见图片后缀
        image_path = None

        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            candidate = os.path.join(
                self.image_dir,
                image_id + ext
            )

            if os.path.isfile(candidate):
                image_path = candidate
                break

        if image_path is None:
            raise FileNotFoundError(
                f"找不到图像: {image_id} "
                f"(目录: {self.image_dir})"
            )

        # 读取图片并统一为 RGB 三通道
        image = Image.open(
            image_path
        ).convert("RGB")

        # DR 分级标签：0~4
        label = int(row["diagnosis"])

        # 数据增强 / 标准化
        if self.transform is not None:
            image = self.transform(image)

        # CrossEntropyLoss 要求类别标签为 long
        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return image, label
