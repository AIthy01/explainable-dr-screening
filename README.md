# Explainable Diabetic Retinopathy Screening and Referral Support

**面向基层医疗的可解释性糖尿病视网膜病变智能筛查与辅助转诊系统**

本项目构建了一套面向基层医疗场景的糖尿病视网膜病变（Diabetic Retinopathy, DR）智能筛查原型系统。系统以眼底彩照为输入，完成：

- ICDR 五级糖尿病视网膜病变严重程度分类
- 基于 DR 严重程度的 Referable DR 筛查
- Grad-CAM 可解释性热力图生成
- Gradio Web 交互式辅助筛查界面
- 面向基层医疗场景的辅助转诊支持

> **Important:** 本项目仅用于科研、教学与技术验证，不用于替代专业眼科医生，也未经过前瞻性临床验证。

---

## 1. Background

糖尿病视网膜病变是糖尿病常见的微血管并发症之一。基层医疗机构在眼底筛查能力、眼科专家资源及患者转诊效率等方面仍面临一定限制。

本项目尝试将深度学习分类、模型融合、风险筛查与可解释人工智能结合，形成从眼底图像输入到辅助转诊提示输出的一体化原型工作流。

系统采用 International Clinical Diabetic Retinopathy（ICDR）五级分级体系：

| Grade | Severity |
|---|---|
| 0 | No DR |
| 1 | Mild NPDR |
| 2 | Moderate NPDR |
| 3 | Severe NPDR |
| 4 | Proliferative DR |

本项目进一步定义**基于 DR 严重程度的 Referable DR（severity-based Referable DR）**：

- `Grade 0–1`：Non-referable DR
- `Grade 2–4`：Referable DR

需要特别强调：真实临床转诊决策通常还需要结合糖尿病黄斑水肿（DME）、视力、其他眼底病变及患者整体临床状况等信息。

APTOS 2019 数据集并不提供完整的上述临床信息，因此，本项目中的 Referable DR 仅表示：

> **基于 DR 严重程度的筛查结果，而非完整临床转诊判定。**

---

## 2. Dataset

模型开发基于 **APTOS 2019 Blindness Detection** 数据集。

本项目使用其中 2930 张带标签眼底图像进行内部开发，并采用分层方式划分：

- Training set: 2344 images
- Validation set: 586 images
- Split ratio: 80% / 20%

训练集类别分布：

| Grade | Number |
|---|---:|
| 0 | 1147 |
| 1 | 240 |
| 2 | 647 |
| 3 | 123 |
| 4 | 187 |

原始数据集不包含在本仓库中。

如需重新训练模型，请自行获取相应数据，并组织为：

```text
data/
├── train.csv
└── train_images/
```

---

## 3. Model Development

本项目分别训练并比较四种基于 ImageNet 预训练的卷积神经网络：

- VGG16
- ResNet50
- DenseNet121
- EfficientNet-B3

主要训练配置如下：

| Parameter | Setting |
|---|---|
| Input size | 224 × 224 |
| Number of classes | 5 |
| Batch size | 16 |
| Epochs | 30 |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Loss | Weighted Cross Entropy |
| Scheduler | ReduceLROnPlateau |
| Random seed | 42 |

为缓解 ICDR 各等级之间明显的类别不平衡，训练过程中采用加权交叉熵损失。

训练集所使用的类别权重为：

```text
[0.4087, 1.9533, 0.7246, 3.8114, 2.5070]
```

---

## 4. Single-model Performance

四个单模型在内部验证集上的主要结果如下：

| Model | Accuracy | Macro-F1 | Macro-AUC |
|---|---:|---:|---:|
| VGG16 | 0.8430 | 0.7403 | 0.9529 |
| ResNet50 | 0.8515 | 0.7346 | 0.9396 |
| DenseNet121 | 0.8464 | 0.7444 | 0.9424 |
| EfficientNet-B3 | 0.8464 | 0.7292 | 0.9462 |

除常规分类指标外，本项目还进一步分析了：

- 模型间错误重叠
- pairwise complementarity
- oracle performance
- 不同类别上的 rescue pattern
- 不同模型组合的筛查互补性

VGG16 与 ResNet50 表现出较好的互补性，因此最终选取二者进行概率级融合。

---

## 5. Ensemble Model

最终采用固定权重的 soft-voting：

```text
P_final = 0.60 × P_VGG16 + 0.40 × P_ResNet50
```

五分类最终内部验证结果：

| Metric | Value |
|---|---:|
| Accuracy | **0.8703** |
| Macro-F1 | **0.7713** |
| Macro-AUC | **0.9568** |

相较单模型，该融合策略进一步提高了整体五分类性能。

---

## 6. Severity-based Referable DR Screening

对每张眼底图像，定义基于严重程度的 Referable DR score：

```text
S_ref = P(Grade 2) + P(Grade 3) + P(Grade 4)
```

开发阶段最终采用阈值：

```text
Threshold = 0.40
```

内部验证结果：

| Metric | Value |
|---|---:|
| ROC-AUC | **0.9896** |
| Recall / Sensitivity | **0.9749** |
| Specificity | **0.9366** |
| Precision | **0.9137** |
| NPV | **0.9819** |
| F1-score | **0.9433** |
| Accuracy | **