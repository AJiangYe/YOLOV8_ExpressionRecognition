import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm


# 配置中文字体
def configure_font():
    font_candidates = ["SimHei", "Microsoft YaHei", "Heiti TC", "WenQuanYi Micro Hei", "Arial Unicode MS"]
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    for font in font_candidates:
        if font in available_fonts:
            plt.rcParams["font.family"] = [font]
            return True
    print("警告: 未找到中文字体，可能导致中文显示异常")
    return False


configure_font()

# 模型与类别配置
MODEL_PATH = "./runs/detect/train3/weights/best.pt"
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOJI_MAP = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😄", "neutral": "😐", "sad": "😢", "surprise": "😲"
}


class ImageProcessingEmotionRecognizer:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        if self.model.task != 'detect':
            raise RuntimeError("请使用YOLOv8检测模型")
        self.class_id_map = {i: cls for i, cls in enumerate(EMOTION_CLASSES)}

    def preprocess_image(self, image):
        """增强型图像预处理，支持任意彩色图片输入"""
        # 1. 确保图像为3通道RGB格式
        if len(image.shape) == 2:  # 灰度图转彩色
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA转RGB
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # 2. 统一调整为模型推荐尺寸(640x640)，保持纵横比
        h, w = image.shape[:2]
        scale = min(640 / w, 640 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h))

        # 3. 创建空白画布并居中放置图像（避免畸变）
        canvas = np.ones((640, 640, 3), dtype=np.uint8) * 255  # 白色背景
        offset_x, offset_y = (640 - new_w) // 2, (640 - new_h) // 2
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

        # 4. 转换为YOLOv8要求的BGR格式
        return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    def detect_emotion(self, image):
        if image is None:
            return None

        try:
            # 增强型预处理
            processed_img = self.preprocess_image(image)

            # 模型推理
            results = self.model(processed_img, conf=0.3)
            if len(results) == 0 or len(results[0].boxes) == 0:
                return None

            # 获取最佳检测结果
            boxes = results[0].boxes
            best_idx = np.argmax(boxes.conf.cpu().numpy())
            best_box = boxes[best_idx]

            return {
                "class_id": int(best_box.cls),
                "confidence": float(best_box.conf),
                "bbox": best_box.xyxy.cpu().numpy().tolist()[0],
                "class_name": self.class_id_map.get(int(best_box.cls), "unknown")
            }
        except Exception as e:
            print(f"检测过程出错: {str(e)}")
            return None

    def visualize_result(self, original_image, detection_result):
        if detection_result is None:
            return "未检测到情绪区域", "🤷", None, None

        # 生成概率分布
        class_id = detection_result["class_id"]
        confidence = detection_result["confidence"]
        probs = np.zeros(len(EMOTION_CLASSES))
        probs[class_id] = confidence
        remaining = (1.0 - confidence) / (len(EMOTION_CLASSES) - 1) if len(EMOTION_CLASSES) > 1 else 0
        for i in range(len(probs)):
            if i != class_id:
                probs[i] = remaining

        # 生成概率直方图
        plt.figure(figsize=(10, 5))
        bars = plt.bar(EMOTION_CLASSES, probs, color='skyblue')
        bars[class_id].set_color('blue')
        plt.title("情绪概率分布")
        plt.xlabel("情绪类别")
        plt.ylabel("置信度")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        hist_path = "emotion_histogram.png"
        plt.savefig(hist_path)
        plt.close()

        # 绘制带边界框的原始图像
        vis_image = original_image.copy()
        bbox = detection_result["bbox"]
        # 将模型输入坐标转换回原始图像坐标
        h, w = original_image.shape[:2]
        scale = min(640 / w, 640 / h)
        offset_x, offset_y = (640 - int(w * scale)) // 2, (640 - int(h * scale)) // 2

        # 还原边界框到原始图像
        x1, y1, x2, y2 = bbox
        x1 = int((x1 - offset_x) / scale)
        y1 = int((y1 - offset_y) / scale)
        x2 = int((x2 - offset_x) / scale)
        y2 = int((y2 - offset_y) / scale)

        # 绘制边界框和标签
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{detection_result['class_name']}: {confidence:.2%}"
        cv2.putText(
            vis_image, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

        # 保存可视化结果
        vis_path = "detection_visualization.png"
        cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

        return (
            f"{detection_result['class_name']} (置信度: {confidence:.2%})",
            EMOJI_MAP[detection_result['class_name']],
            vis_path,
            hist_path
        )

    def create_interface(self):
        with gr.Blocks(title="人脸情绪检测系统") as demo:
            gr.Markdown("""
            # 😊 人脸情绪检测系统
            ### 支持任意彩色图片输入的情绪识别
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="numpy", label="上传图像（支持任意格式）")
                    detect_btn = gr.Button("开始检测", variant="primary")

                with gr.Column(scale=2):
                    with gr.Row():
                        result_label = gr.Label(label="检测结果")
                        emoji_display = gr.Textbox(label="情绪表情", interactive=False)
                    with gr.Row():
                        with gr.Column():
                            detection_vis = gr.Image(label="检测可视化", type="filepath")
                        with gr.Column():
                            probability_hist = gr.Image(label="情绪概率分布", type="filepath")

            def process_image(image):
                if image is None:
                    return "请上传图像", "❓", None, None

                try:
                    detection_result = self.detect_emotion(image)
                    result_text, emoji, vis_path, hist_path = self.visualize_result(image, detection_result)
                    return result_text, emoji, vis_path, hist_path
                except Exception as e:
                    return f"处理错误: {str(e)}", "💥", None, None

            detect_btn.click(
                fn=process_image,
                inputs=input_image,
                outputs=[result_label, emoji_display, detection_vis, probability_hist]
            )

            input_image.change(
                fn=process_image,
                inputs=input_image,
                outputs=[result_label, emoji_display, detection_vis, probability_hist]
            )

        return demo


if __name__ == "__main__":
    try:
        app = ImageProcessingEmotionRecognizer()
        demo = app.create_interface()
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    except Exception as e:
        print(f"应用启动失败: {str(e)}")