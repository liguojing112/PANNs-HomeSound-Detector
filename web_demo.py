"""Gradio web application for home sound detection."""

from __future__ import annotations

import argparse
from typing import Tuple

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd

from inference import PannsPredictor
from utils import setup_logging


def build_plot(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(df["label"][::-1], df["confidence"][::-1], color="#2a7fff")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Top-5 predictions")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    return fig


def make_infer_fn(predictor: PannsPredictor):
    def infer(audio_path: str) -> Tuple[pd.DataFrame, str, object]:
        if not audio_path:
            raise gr.Error("请先上传音频，或使用麦克风录制一段不超过 10 秒的声音。")

        result = predictor.predict_file(audio_path, top_k=5)
        df = pd.DataFrame(
            [{"label": item.label, "confidence": round(item.score, 4)} for item in result.predictions]
        )
        summary = (
            f"时长: {result.duration_sec:.2f}s | "
            f"窗口数: {result.used_windows} | "
            f"推理耗时: {result.elapsed_ms:.2f} ms | "
            f"设备: {predictor.device.upper()}"
        )
        fig = build_plot(df)
        return df, summary, fig

    return infer


def main() -> None:
    parser = argparse.ArgumentParser(description="Gradio demo for the home sound detector")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional local PANNs checkpoint path")
    parser.add_argument("--device", type=str, default=None, help="Force device, e.g. cpu or cuda")
    parser.add_argument("--server_port", type=int, default=7860, help="Gradio server port")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    args = parser.parse_args()

    setup_logging(args.log_level)
    predictor = PannsPredictor(checkpoint_path=args.checkpoint, device=args.device)

    with gr.Blocks(title="日常环境声音检测系统") as demo:
        gr.Markdown(
            """
            # 日常环境声音检测系统
            上传音频文件或直接录音，系统会使用 PANNs Cnn14 输出 Top-5 环境声音类别和置信度。
            超过 10 秒的音频会自动按 5 秒窗口、2.5 秒步长进行切片并汇总结果。
            """
        )
        with gr.Row():
            audio_input = gr.Audio(
                label="上传音频 / 麦克风录制",
                type="filepath",
                sources=["upload", "microphone"],
            )
        infer_btn = gr.Button("识别", variant="primary")
        result_table = gr.Dataframe(
            headers=["label", "confidence"],
            datatype=["str", "number"],
            label="识别结果",
            interactive=False,
        )
        summary_box = gr.Textbox(label="推理摘要", interactive=False)
        plot_output = gr.Plot(label="置信度柱状图")
        infer_btn.click(
            fn=make_infer_fn(predictor),
            inputs=[audio_input],
            outputs=[result_table, summary_box, plot_output],
        )

    demo.launch(server_port=args.server_port, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
