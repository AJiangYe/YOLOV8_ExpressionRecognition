import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 配置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 模型与类别配置
MODEL_PATH = "./runs/detect/train3/weights/best.pt"
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOJI_MAP = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😄", "neutral": "😐", "sad": "😢", "surprise": "😲"
}


class EmotionRecognitionApp:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)

    def predict_emotion(self, image):
        # 图像预处理
        if image is None:
            return None, None, None
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        results = self.model(img)
        probs = results[0].probs.data.cpu().numpy()
        max_idx = np.argmax(probs)

        # 生成直方图
        plt.figure(figsize=(10, 5))
        bars = plt.bar(EMOTION_CLASSES, probs, color='skyblue')
        bars[max_idx].set_color('blue')
        plt.title("情绪概率分布")
        plt.xlabel("情绪类别")
        plt.ylabel("概率值")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        hist_path = "emotion_histogram.png"
        plt.savefig(hist_path)
        plt.close()

        return (
            f"{EMOTION_CLASSES[max_idx]} (概率: {probs[max_idx]:.2%})",
            EMOJI_MAP[EMOTION_CLASSES[max_idx]],
            hist_path
        )

    def create_interface(self):
        with gr.Blocks(title="人脸情绪识别系统") as demo:
            gr.Markdown("# 😊 人脸情绪识别系统")
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="numpy", label="上传图像", shape=(64, 64))
                    submit_btn = gr.Button("开始识别", variant="primary")

                with gr.Column(scale=2):
                    with gr.Row():
                        result_label = gr.Label(label="识别结果", font_size=24)
                        emoji_display = gr.Textbox(label="情绪表情", font_size=48, interactive=False)
                    with gr.Row():
                        histogram = gr.Image(label="概率分布", type="filepath")

            submit_btn.click(
                fn=self.predict_emotion,
                inputs=input_image,
                outputs=[result_label, emoji_display, histogram]
            )

            input_image.change(
                fn=self.predict_emotion,
                inputs=input_image,
                outputs=[result_label, emoji_display, histogram]
            )

        return demo


if __name__ == "__main__":
    app = EmotionRecognitionApp()
    demo = app.create_interface()
    demo.launch(share=True,server_name="0.0.0.0", server_port=7860)