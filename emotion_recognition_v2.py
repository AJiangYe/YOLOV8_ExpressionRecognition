import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 配置中文显示
plt.rcParams["font.family"] = ["SimHei", "SimHei", "SimHei"]

# 模型与类别配置（适配检测模型）
MODEL_PATH = "./runs/detect/train3/weights/best.pt"  # 检测模型路径
EMOTION_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]  # 7类情绪
EMOJI_MAP = {
    "angry": "😠", "disgust": "🤢", "fear": "😨",
    "happy": "😄", "neutral": "😐", "sad": "😢", "surprise": "😲"
}


class DetectionToClassificationAdapter:
    def __init__(self):
        # 加载检测模型
        self.model = YOLO(MODEL_PATH)
        # 验证模型类型
        if self.model.task != 'detect':
            raise RuntimeError("请使用YOLOv8检测模型")

        # 类别ID映射（确保与检测模型的classes.txt一致）
        self.class_id_map = {i: cls for i, cls in enumerate(EMOTION_CLASSES)}

    def detect_emotion(self, image):
        """使用检测模型进行情绪识别（提取置信度最高的目标）"""
        if image is None:
            return None

        # 转换为BGR格式（YOLOv8默认输入）
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 模型推理（检测人脸情绪区域）
        results = self.model(image, conf=0.3)  # 置信度阈值0.3

        # 提取置信度最高的检测结果
        if len(results) == 0 or len(results[0].boxes) == 0:
            return None

        # 获取最佳检测框
        boxes = results[0].boxes
        best_idx = np.argmax(boxes.conf.cpu().numpy())  # 最高置信度索引
        best_box = boxes[best_idx]

        return {
            "class_id": int(best_box.cls),
            "confidence": float(best_box.conf),
            "bbox": best_box.xyxy.cpu().numpy().tolist()[0],  # [x1,y1,x2,y2]
            "class_name": self.class_id_map.get(int(best_box.cls), "unknown")
        }

    def visualize_result(self, image, detection_result):
        """可视化检测结果和情绪概率"""
        if detection_result is None:
            return "未检测到情绪区域", "🤷", None

        # 创建情绪概率分布（模拟分类模型输出）
        # 检测模型只能提供单个类别置信度，这里将其转换为概率分布
        class_id = detection_result["class_id"]
        confidence = detection_result["confidence"]
        probs = np.zeros(len(EMOTION_CLASSES))
        probs[class_id] = confidence
        # 分配剩余概率给其他类别（模拟分布）
        remaining = (1.0 - confidence) / (len(EMOTION_CLASSES) - 1)
        for i in range(len(probs)):
            if i != class_id:
                probs[i] = remaining

        # 生成概率直方图
        plt.figure(figsize=(10, 5))
        bars = plt.bar(EMOTION_CLASSES, probs, color='skyblue')
        bars[class_id].set_color('blue')  # 高亮检测类别
        plt.title("情绪概率分布（检测模型模拟）")
        plt.xlabel("情绪类别")
        plt.ylabel("置信度")
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        hist_path = "emotion_detection_histogram.png"
        plt.savefig(hist_path)
        plt.close()

        # 绘制带边界框的图像
        vis_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) if len(image.shape) == 3 else image
        bbox = detection_result["bbox"]
        x1, y1, x2, y2 = map(int, bbox)
        # 绘制边界框
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 添加类别标签
        label = f"{detection_result['class_name']}: {confidence:.2%}"
        cv2.putText(
            vis_image, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )
        # 保存可视化结果
        vis_path = "detection_visualization.png"
        cv2.imwrite(vis_path, vis_image)

        return (
            f"{detection_result['class_name']} (置信度: {confidence:.2%})",
            EMOJI_MAP[detection_result['class_name']],
            hist_path
        )

    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="人脸情绪检测系统") as demo:
            gr.Markdown("""
            # 😊 人脸情绪检测系统
            ### 基于YOLOv8检测模型的情绪识别
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="numpy", label="上传图像")
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
                    # 检测情绪区域
                    detection_result = self.detect_emotion(image)
                    # 可视化结果
                    result_text, emoji, hist_path = self.visualize_result(image, detection_result)
                    # 返回检测可视化图像
                    return result_text, emoji, "detection_visualization.png", hist_path
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
        app = DetectionToClassificationAdapter()
        demo = app.create_interface()
        demo.launch(server_name="127.0.0.1", server_port=7860)
    except Exception as e:
        print(f"应用启动失败: {str(e)}")