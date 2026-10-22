import os
import uuid
import json
from pathlib import Path

import numpy as np
import gradio as gr
from PIL import Image

from predictor import DRPredictor
from ensemble_gradcam import explain_image


# ============================================================
# Frozen research configuration
# ============================================================

CLASS_NAMES = {
    0: "No DR",
    1: "Mild NPDR",
    2: "Moderate NPDR",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

CLASS_NAMES_CN = {
    0: "无糖尿病视网膜病变",
    1: "轻度非增殖期 DR",
    2: "中度非增殖期 DR",
    3: "重度非增殖期 DR",
    4: "增殖期 DR",
}

VGG_WEIGHT = 0.60
RESNET_WEIGHT = 0.40
REFERRAL_THRESHOLD = 0.40

RUNTIME_DIR = Path(
    "results/runtime/gradio"
)

INPUT_DIR = (
    RUNTIME_DIR / "inputs"
)

CAM_DIR = (
    RUNTIME_DIR / "gradcam"
)

INPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CAM_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Load frozen inference system exactly once
# ============================================================

print("=" * 80)
print("Initializing DR screening system...")
print("=" * 80)

PREDICTOR = DRPredictor()

print("=" * 80)
print("System ready.")
print("=" * 80)


# ============================================================
# Utilities
# ============================================================

def normalize_probability_dict(
    class_probabilities
):
    """
    Convert predictor output to:
        {
            "0 | No DR": score,
            ...
        }

    Handles list / ndarray / dict robustly.
    """

    if isinstance(
        class_probabilities,
        dict
    ):
        values = []

        for grade in range(5):

            candidates = [
                grade,
                str(grade),
                CLASS_NAMES[grade],
                f"grade_{grade}_{CLASS_NAMES[grade]}",
            ]

            value = None

            for key in candidates:
                if key in class_probabilities:
                    value = class_probabilities[key]
                    break

            if value is None:
                raise KeyError(
                    "Cannot interpret "
                    "class_probabilities dict: "
                    f"{class_probabilities}"
                )

            values.append(
                float(value)
            )

    else:
        values = np.asarray(
            class_probabilities,
            dtype=float,
        ).reshape(-1)

        if len(values) != 5:
            raise ValueError(
                "Expected five class "
                "probabilities, got "
                f"{len(values)}"
            )

    return {
        (
            f"Grade {grade} | "
            f"{CLASS_NAMES[grade]}"
        ): float(values[grade])
        for grade in range(5)
    }


def get_result_value(
    result,
    candidates,
    default=None,
):
    """
    Allow small differences in predictor.py key naming.
    """

    for key in candidates:
        if key in result:
            return result[key]

    return default


def save_uploaded_image(
    image_path
):
    """
    Copy uploaded image into a stable runtime location.
    Ensures RGB PNG and a unique filename.
    """

    uid = uuid.uuid4().hex

    output_path = (
        INPUT_DIR
        / f"{uid}.png"
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    image.save(
        output_path
    )

    return str(output_path)


# ============================================================
# Main inference function
# ============================================================

def analyze_fundus(
    image_path
):

    if image_path is None:
        raise gr.Error(
            "请先上传一张眼底彩照。"
        )

    try:
        # ----------------------------------------------------
        # Stable copy of Gradio upload
        # ----------------------------------------------------

        stable_path = save_uploaded_image(
            image_path
        )

        # ----------------------------------------------------
        # Frozen predictor
        # ----------------------------------------------------

        result = PREDICTOR.predict(
            stable_path
        )

        predicted_grade = int(
            get_result_value(
                result,
                [
                    "predicted_grade",
                    "grade",
                ],
            )
        )

        predicted_label = (
            get_result_value(
                result,
                [
                    "predicted_label",
                    "label",
                ],
                CLASS_NAMES[
                    predicted_grade
                ],
            )
        )

        top_score = float(
            get_result_value(
                result,
                [
                    "top_class_score",
                    "top_score",
                    "confidence",
                ],
                np.nan,
            )
        )

        class_probabilities = (
            get_result_value(
                result,
                [
                    "class_probabilities",
                    "probabilities",
                    "fused_probabilities",
                    "fused_probs",
                ],
            )
        )

        if class_probabilities is None:
            raise KeyError(
                "Predictor result does not "
                "contain class probabilities."
            )

        score_dict = (
            normalize_probability_dict(
                class_probabilities
            )
        )

        referable_score = float(
            get_result_value(
                result,
                [
                    "referable_dr_score",
                    "referable_score",
                    "referral_score",
                ],
            )
        )

        referral_triggered = (
            referable_score
            >= REFERRAL_THRESHOLD
        )

        # ----------------------------------------------------
        # Result text
        # ----------------------------------------------------

        grade_markdown = f"""
### DR 严重程度分级

**Grade {predicted_grade} — {predicted_label}**

中文解释：**{CLASS_NAMES_CN[predicted_grade]}**

五分类最高得分：**{top_score:.4f}**

> 严重程度由融合后的五分类分数取 `argmax` 得到。
"""

        if referral_triggered:

            referral_markdown = f"""
### 🔶 辅助转诊提示

**达到本系统的 Referable DR 筛查阈值**

Referable DR score：

**{referable_score:.4f}**

开发阶段工作阈值：

**{REFERRAL_THRESHOLD:.2f}**

本系统中的 Referable DR 定义为 **Grade ≥ 2**
（Moderate NPDR / Severe NPDR / Proliferative DR）。

**建议结合临床情况进一步接受眼科专业评估。**

> 该结果为辅助筛查提示，并非临床诊断或正式转诊决定。
"""

        else:

            referral_markdown = f"""
### 🟢 辅助转诊提示

**当前未达到本系统的 Referable DR 筛查阈值**

Referable DR score：

**{referable_score:.4f}**

开发阶段工作阈值：

**{REFERRAL_THRESHOLD:.2f}**

> 未触发提示不代表排除糖尿病视网膜病变或其他眼底疾病。
> 最终判断仍应结合规范眼科检查。
"""

        # ----------------------------------------------------
        # Grad-CAM
        #
        # Both models explain the SAME final ensemble class.
        # ----------------------------------------------------

        explain_image(
            image_path=stable_path,
            save_dir=str(CAM_DIR),
            target_class=predicted_grade,
            predictor=PREDICTOR,
        )

        stem = Path(
            stable_path
        ).stem

        case_dir = (
            CAM_DIR / stem
        )

        vgg_cam = str(
            case_dir
            / "vgg16_gradcam.png"
        )

        resnet_cam = str(
            case_dir
            / "resnet50_gradcam.png"
        )

        weighted_cam = str(
            case_dir
            / "weighted_cam_summary.png"
        )

        for path in [
            vgg_cam,
            resnet_cam,
            weighted_cam,
        ]:
            if not os.path.isfile(
                path
            ):
                raise FileNotFoundError(
                    f"Grad-CAM output missing: "
                    f"{path}"
                )

        # ----------------------------------------------------
        # Detailed technical information
        # ----------------------------------------------------

        technical_info = {
            "predicted_grade":
                predicted_grade,

            "predicted_label":
                predicted_label,

            "top_class_score":
                round(
                    top_score,
                    6,
                ),

            "referable_dr_score":
                round(
                    referable_score,
                    6,
                ),

            "development_referral_threshold":
                REFERRAL_THRESHOLD,

            "referral_triggered":
                bool(
                    referral_triggered
                ),

            "ensemble": {
                "VGG16":
                    VGG_WEIGHT,

                "ResNet50":
                    RESNET_WEIGHT,
            },

            "gradcam_target_class":
                predicted_grade,

            "gradcam_layers": {
                "VGG16":
                    "conv5_2 / features[26]",

                "ResNet50":
                    "layer4[-1]",
            },

            "class_scores": {
                key:
                    round(
                        value,
                        6,
                    )
                for key, value
                in score_dict.items()
            },

            "interpretability_note":
                (
                    "Grad-CAM indicates image "
                    "regions contributing to the "
                    "classification output. It is "
                    "not a lesion segmentation map."
                ),

            "weighted_cam_note":
                (
                    "The weighted CAM summary is "
                    "an auxiliary visualization "
                    "constructed from normalized "
                    "branch CAMs. It is not a "
                    "mathematically unique "
                    "ensemble Grad-CAM."
                ),
        }

        return (
            grade_markdown,
            referral_markdown,
            score_dict,
            vgg_cam,
            resnet_cam,
            weighted_cam,
            technical_info,
        )

    except Exception as exc:

        import traceback

        traceback.print_exc()

        raise gr.Error(
            "分析失败："
            f"{type(exc).__name__}: "
            f"{exc}"
        )


# ============================================================
# Gradio UI
# ============================================================

APP_CSS = """
.gradio-container {
    max-width: 1320px !important;
    margin: 0 auto !important;
    padding-bottom: 48px !important;
}

body {
    background:
        radial-gradient(circle at 20% 0%, rgba(236, 245, 255, 0.95), transparent 36%),
        linear-gradient(180deg, #f7f9fc 0%, #ffffff 38%);
}

/* Hero */
#dr-hero {
    border-radius: 24px;
    padding: 34px 38px;
    margin: 18px 0 24px 0;
    background:
        linear-gradient(135deg, #102a43 0%, #174a6e 55%, #177e89 100%);
    color: white;
    box-shadow: 0 14px 40px rgba(16, 42, 67, 0.14);
}

#dr-hero h1 {
    margin: 0 0 10px 0;
    color: white;
    font-size: 32px;
    line-height: 1.25;
}

#dr-hero .subtitle {
    font-size: 17px;
    opacity: 0.92;
    margin-bottom: 20px;
}

.hero-chip {
    display: inline-block;
    margin: 5px 7px 0 0;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.13);
    border: 1px solid rgba(255,255,255,0.20);
    font-size: 13px;
}

/* Main cards */
.panel-card {
    background: rgba(255,255,255,0.96);
    border: 1px solid #e6ebf1;
    border-radius: 18px;
    padding: 18px !important;
    box-shadow: 0 8px 26px rgba(15, 42, 66, 0.055);
}

.section-card {
    background: rgba(255,255,255,0.98);
    border: 1px solid #e6ebf1;
    border-radius: 18px;
    padding: 18px 20px !important;
    margin-top: 8px;
    box-shadow: 0 6px 22px rgba(15, 42, 66, 0.045);
}

.section-title {
    margin-top: 28px;
    margin-bottom: 10px;
}

/* Primary button */
#analyze-btn {
    min-height: 52px;
    font-size: 17px;
    font-weight: 700;
    border-radius: 12px !important;
}

/* Result cards */
#grade-result,
#referral-result {
    min-height: 185px;
}

#grade-result h3,
#referral-result h3 {
    margin-top: 2px;
}

/* Images */
.cam-card img {
    border-radius: 12px !important;
}

/* Small info box */
.info-note {
    padding: 13px 15px;
    border-radius: 12px;
    background: #f6f9fc;
    border-left: 4px solid #2f80a3;
    line-height: 1.7;
    font-size: 14px;
}

.warning-note {
    padding: 13px 15px;
    border-radius: 12px;
    background: #fff9ef;
    border-left: 4px solid #e7a93b;
    line-height: 1.7;
    font-size: 14px;
}

/* Model summary */
.model-box {
    margin-top: 18px;
    padding: 22px 24px;
    border-radius: 16px;
    background: #f8fafc;
    border: 1px solid #e3e8ef;
}

.model-formula {
    font-family: "Times New Roman", serif;
    font-size: 25px;
    margin: 12px 0 18px 0;
}

.footer-note {
    margin-top: 26px;
    text-align: center;
    color: #6b7785;
    font-size: 13px;
    line-height: 1.7;
}

/* Mobile */
@media (max-width: 720px) {
    #dr-hero {
        padding: 24px 21px;
        border-radius: 18px;
    }

    #dr-hero h1 {
        font-size: 25px;
    }

    .panel-card,
    .section-card {
        padding: 14px !important;
    }
}


/* =========================================================
   UI polish v2
   ========================================================= */

/* Hero: improve contrast */
#dr-hero,
#dr-hero * {
    color: #ffffff !important;
}

#dr-hero .subtitle {
    color: rgba(255, 255, 255, 0.92) !important;
    font-weight: 500;
}

#dr-hero .hero-chip {
    color: #ffffff !important;
    background: rgba(255, 255, 255, 0.14) !important;
    border: 1px solid rgba(255, 255, 255, 0.28) !important;
    backdrop-filter: blur(4px);
}

/* Stronger visual hierarchy */
.section-title h2 {
    font-size: 25px !important;
    letter-spacing: -0.3px;
}

.panel-card,
.section-card {
    transition:
        box-shadow 0.18s ease,
        transform 0.18s ease;
}

/* Main result cards */
#grade-result {
    border-left: 5px solid #277da1;
}

#referral-result {
    border-left: 5px solid #f28c28;
}

/* Primary button */
#analyze-btn {
    background:
        linear-gradient(
            90deg,
            #f47b20 0%,
            #f59e32 100%
        ) !important;

    border: none !important;

    box-shadow:
        0 7px 18px rgba(244, 123, 32, 0.22);
}

#analyze-btn:hover {
    filter: brightness(1.04);
}

/* Better image presentation */
.cam-card {
    min-width: 0;
}

.cam-card img {
    object-fit: contain !important;
    background: #0b1320;
}

/* Accordion refinement */
.gradio-accordion {
    border-radius: 12px !important;
}

/* Better footer separation */
.footer-note {
    border-top: 1px solid #e7ebf0;
    padding-top: 20px;
}

/* Print / Save-as-PDF optimization */
@media print {

    body {
        background: #ffffff !important;
    }

    .gradio-container {
        max-width: 100% !important;
        padding: 0 !important;
    }

    #dr-hero {
        box-shadow: none !important;
        break-inside: avoid;
    }

    .panel-card,
    .section-card,
    .model-box,
    .info-note,
    .warning-note {
        box-shadow: none !important;
        break-inside: avoid;
    }

    .cam-card {
        break-inside: avoid;
    }

    body::before,
    body::after {
        display: none !important;
        content: none !important;
    }

    * {
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }
}

"""


with gr.Blocks(
    title="可解释糖尿病视网膜病变智能筛查与辅助转诊系统",
    css=APP_CSS,
) as demo:

    # --------------------------------------------------------
    # Hero
    # --------------------------------------------------------

    gr.HTML(
        """
        <div id="dr-hero">
            <h1>👁️ 可解释糖尿病视网膜病变智能筛查与辅助转诊系统</h1>

            <div class="subtitle">
                面向基层医疗场景的 DR 智能筛查研究原型
            </div>

            <span class="hero-chip">5级 DR 严重程度</span>
            <span class="hero-chip">Referable DR 筛查</span>
            <span class="hero-chip">VGG16 + ResNet50</span>
            <span class="hero-chip">Grad-CAM 可解释性</span>
        </div>
        """
    )

    gr.HTML(
        """
        <div class="info-note">
            <b>使用说明：</b>
            上传一张完整、清晰的眼底彩色照片。系统将给出 DR 严重程度、
            五分类融合分数、Referable DR score、辅助转诊提示及可解释性结果。
            图像按照与模型验证阶段一致的预处理流程进入推理。
        </div>
        """
    )

    # --------------------------------------------------------
    # Input + main clinical-style result
    # --------------------------------------------------------

    with gr.Row():

        with gr.Column(
            scale=1,
            elem_classes=["panel-card"],
        ):

            gr.Markdown("### ① 上传眼底图像")

            input_image = gr.Image(
                label="眼底彩照",
                type="filepath",
                image_mode="RGB",
            )

            analyze_button = gr.Button(
                "开始智能筛查",
                variant="primary",
                elem_id="analyze-btn",
            )

            gr.HTML(
                """
                <div class="warning-note">
                    本系统是科研与教学原型。
                    请勿上传包含可识别个人身份信息的医疗图像。
                </div>
                """
            )

        with gr.Column(
            scale=1,
            elem_classes=["panel-card"],
        ):

            gr.Markdown("### ② 核心筛查结果")

            with gr.Column(
                elem_classes=["section-card"],
                elem_id="grade-result",
            ):
                grade_output = gr.Markdown(
                    """
### DR 严重程度

等待图像分析……
"""
                )

            with gr.Column(
                elem_classes=["section-card"],
                elem_id="referral-result",
            ):
                referral_output = gr.Markdown(
                    """
### 辅助转诊提示

等待图像分析……
"""
                )

    # --------------------------------------------------------
    # Scores
    # --------------------------------------------------------

    gr.Markdown(
        "## ③ 五分类融合分数",
        elem_classes=["section-title"],
    )

    with gr.Column(
        elem_classes=["section-card"]
    ):

        gr.HTML(
            """
            <div class="info-note">
                分数来自冻结后的 VGG16 0.60 + ResNet50 0.40 soft-voting 融合。
                这些数值用于模型内部分类比较，不应直接解释为经过临床校准的疾病概率。
            </div>
            """
        )

        score_output = gr.Label(
            label="DR 五分类融合分数",
            num_top_classes=5,
        )

    # --------------------------------------------------------
    # Explainability
    # --------------------------------------------------------

    gr.Markdown(
        "## ④ 可解释性结果",
        elem_classes=["section-title"],
    )

    with gr.Column(
        elem_classes=["section-card"]
    ):

        gr.HTML(
            """
            <div class="info-note">
                两个 Grad-CAM 分支均针对
                <b>最终融合模型预测出的同一个类别</b>生成。
                热区表示对该类别输出贡献较大的图像区域，
                <b>不等同于病灶定位、病灶分割或临床诊断依据。</b>
            </div>
            """
        )

        with gr.Row():

            with gr.Column(
                elem_classes=["cam-card"]
            ):
                vgg_output = gr.Image(
                    label="VGG16 Grad-CAM · conv5_2",
                    type="filepath",
                )

            with gr.Column(
                elem_classes=["cam-card"]
            ):
                resnet_output = gr.Image(
                    label="ResNet50 Grad-CAM · layer4[-1]",
                    type="filepath",
                )

        with gr.Accordion(
            "实验性辅助可视化 · Weighted CAM Summary",
            open=False,
        ):

            gr.HTML(
                """
                <div class="warning-note">
                    此图由两个归一化分支 CAM 按 0.60 / 0.40
                    进行视觉汇总，仅用于观察空间响应。
                    它不是严格定义的 ensemble Grad-CAM，
                    也不能理解为两个模型共同确认的病灶。
                </div>
                """
            )

            weighted_output = gr.Image(
                label="Weighted CAM Summary",
                type="filepath",
            )

    # --------------------------------------------------------
    # Technical information
    # --------------------------------------------------------

    gr.Markdown(
        "## ⑤ 系统与技术信息",
        elem_classes=["section-title"],
    )

    with gr.Column(
        elem_classes=["section-card"]
    ):

        gr.HTML(
            """
            <div class="model-box">

                <div style="font-size:18px;font-weight:700;margin-bottom:16px;">
                    冻结后的最终配置
                </div>

                <div>
                    <b>分类融合：</b>
                    VGG16 0.60 + ResNet50 0.40
                </div>

                <div style="margin-top:18px;">
                    <b>Referable DR score</b>
                </div>

                <div class="model-formula">
                    <i>S</i><sub>ref</sub>
                    =
                    <i>P</i>(Grade 2)
                    +
                    <i>P</i>(Grade 3)
                    +
                    <i>P</i>(Grade 4)
                </div>

                <div>
                    <b>开发阶段辅助筛查阈值：</b>
                </div>

                <div class="model-formula">
                    <i>t</i><sub>ref</sub> = 0.40
                </div>

                <div style="line-height:1.8;">
                    VGG16 Grad-CAM：
                    <b>conv5_2 / features[26]</b><br>

                    ResNet50 Grad-CAM：
                    <b>layer4[-1]</b>
                </div>

                <div style="margin-top:18px;color:#8a5b15;">
                    阈值来源于当前内部开发/验证集分析，
                    <b>尚未经过独立外部临床验证。</b>
                </div>

            </div>
            """
        )

        with gr.Accordion(
            "查看本次推理的详细技术信息",
            open=False,
        ):
            technical_output = gr.JSON(
                label="Inference metadata"
            )

    # --------------------------------------------------------
    # Clinical / research limitation
    # --------------------------------------------------------

    gr.HTML(
        """
        <div class="warning-note" style="margin-top:28px;">
            <b>重要声明：</b>
            本系统基于 APTOS 2019 数据构建，
            用于课程、科研中的算法验证与原型展示。
            系统输出不能替代眼科医师诊断，
            不能作为独立临床转诊依据。
            未触发 Referable DR 提示也不能排除其他眼底疾病。
        </div>

        <div class="footer-note">
            Explainable Diabetic Retinopathy Screening Prototype
            · Research Use Only
        </div>
        """
    )

    # --------------------------------------------------------
    # Event
    # --------------------------------------------------------

    analyze_button.click(
        fn=analyze_fundus,
        inputs=[
            input_image
        ],
        outputs=[
            grade_output,
            referral_output,
            score_output,
            vgg_output,
            resnet_output,
            weighted_output,
            technical_output,
        ],
    )


# ============================================================
# Launch
# ============================================================

# ============================================================
# Web authentication
# ============================================================

AUTH_USER = os.environ.get("DR_WEB_USER")
AUTH_PASSWORD = os.environ.get("DR_WEB_PASSWORD")

if not AUTH_USER or not AUTH_PASSWORD:
    raise RuntimeError(
        "DR_WEB_USER / DR_WEB_PASSWORD are not configured."
    )


if __name__ == "__main__":

    demo.queue()

    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        auth=(AUTH_USER, AUTH_PASSWORD),
        auth_message=(
            "该系统为科研与教学演示原型。"
            "请输入授权账号和密码后访问。"
        ),
        show_error=True,
    )
