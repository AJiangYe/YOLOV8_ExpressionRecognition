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


class ROIOptimizedEmotionRecognizer:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        if self.model.task != 'detect':
            raise RuntimeError("请使用YOLOv8检测模型")
        self.class_id_map = {i: cls for i, cls in enumerate(EMOTION_CLASSES)}

        # 优化参数
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        self.min_face_size = (80, 80)  # 最小人脸尺寸
        self.conf_threshold = 0.35  # 置信度阈值
        self.nms_threshold = 0.45  # 非极大值抑制阈值
        self.roi_expansion = 0.2  # 区域扩展系数

    def precise_face_detection(self, image):
        """精确人脸区域检测，结合多特征验证"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 多级人脸检测
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=8,
            minSize=self.min_face_size, flags=cv2.CASCADE_SCALE_IMAGE
        )

        if len(faces) == 0:
            return None

        # 筛选最佳人脸区域（最大且包含眼睛）
        best_face = None
        max_score = 0

        for (x, y, w, h) in faces:
            # 扩展人脸区域
            h, w_img = image.shape[:2]
            x1 = max(0, int(x - w * self.roi_expansion))
            y1 = max(0, int(y - h * self.roi_expansion))
            x2 = min(w_img, int(x + w + w * self.roi_expansion))
            y2 = min(h, int(y + h + h * self.roi_expansion))
            face_roi = image[y1:y2, x1:x2]

            # 眼睛检测验证
            roi_gray = cv2.cvtColor(face_roi, cv2.COLOR_RGB2GRAY)
            eyes = self.eye_detector.detectMultiScale(roi_gray, minSize=(20, 20))

            # 评分机制：大小+眼睛数量
            score = (w * h) + (len(eyes) * 1000)
            if score > max_score:
                max_score = score
                best_face = {
                    "roi": face_roi,
                    "original_coords": (x1, y1, x2, y2)
                }

        return best_face

    def preprocess_image(self, image):
        """优化的图像预处理流程"""
        # 1. 精确人脸定位
        face_data = self.precise_face_detection(image)
        if face_data is None:
            # 回退到通用预处理
            processed_img = self.generic_preprocess(image)
            return processed_img, None
        else:
            # 使用检测到的人脸区域
            processed_img = self.face_specific_preprocess(face_data["roi"])
            return processed_img, face_data["original_coords"]

    def face_specific_preprocess(self, face_roi):
        """针对人脸区域的预处理"""
        # 保持纵横比调整大小
        h, w = face_roi.shape[:2]
        scale = min(640 / w, 640 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(face_roi, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 创建居中画布
        canvas = np.ones((640, 640, 3), dtype=np.uint8) * 255
        offset_x, offset_y = (640 - new_w) // 2, (640 - new_h) // 2
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

        return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    def generic_preprocess(self, image):
        """通用图像预处理（无检测到人脸时使用）"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        h, w = image.shape[:2]
        scale = min(640 / w, 640 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.ones((640, 640, 3), dtype=np.uint8) * 255
        offset_x, offset_y = (640 - new_w) // 2, (640 - new_h) // 2
        canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

        return cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

    def detect_emotion(self, image):
        if image is None:
            return None, None

        try:
            processed_img, face_coords = self.preprocess_image(image)
            results = self.model(
                processed_img,
                conf=self.conf_threshold,
                iou=self.nms_threshold,
                agnostic_nms=True
            )

            if len(results) == 0 or len(results[0].boxes) == 0:
                return None, face_coords

            boxes = results[0].boxes
            best_idx = np.argmax(boxes.conf.cpu().numpy())
            best_box = boxes[best_idx]

            return {
                "class_id": int(best_box.cls),
                "confidence": float(best_box.conf),
                "bbox": best_box.xyxy.cpu().numpy().tolist()[0],
                "class_name": self.class_id_map.get(int(best_box.cls), "unknown")
            }, face_coords
        except Exception as e:
            print(f"检测错误: {str(e)}")
            return None, None

    def visualize_result(self, original_image, detection_result, face_coords):
        if detection_result is None:
            return "未检测到有效人脸", "🤷", None, None

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

        # 绘制可视化结果
        vis_image = original_image.copy()

        # 绘制人脸区域框
        if face_coords:
            x1, y1, x2, y2 = face_coords
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (255, 165, 0), 2)  # 橙色框
            cv2.putText(
                vis_image, "人脸区域", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2
            )

        # 绘制情绪检测框
        bbox = detection_result["bbox"]
        h, w = vis_image.shape[:2]
        x1, y1, x2, y2 = [int(coord) for coord in bbox]

        # 转换坐标到原始图像
        scale = min(w / 640, h / 640)
        x1 = int(x1 * scale)
        y1 = int(y1 * scale)
        x2 = int(x2 * scale)
        y2 = int(y2 * scale)

        cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{detection_result['class_name']}: {confidence:.2%}"
        cv2.putText(
            vis_image, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2
        )

        vis_path = "detection_visualization.png"
        cv2.imwrite(vis_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

        return (
            f"{detection_result['class_name']} (置信度: {confidence:.2%})",
            EMOJI_MAP[detection_result['class_name']],
            vis_path,
            hist_path
        )

    def create_interface(self):
        with gr.Blocks(title="人脸情绪检测系统_V4") as demo:
            gr.Markdown("""
            # 😊 人脸情绪检测系统
            ### 精准人脸区域定位与表情识别
            """)
            with gr.Row():
                with gr.Column(scale=1):
                    input_image = gr.Image(type="numpy", label="上传图像")
                    detect_btn = gr.Button("开始检测", variant="primary")
                    gr.Markdown("""
                    **优化特点：**
                    - 多级人脸检测与验证
                    - 眼睛特征辅助定位
                    - 自适应区域扩展
                    - 非极大值抑制去重
                    """)

                with gr.Column(scale=2):
                    with gr.Row():
                        result_label = gr.Label(label="识别结果")
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
                    detection_result, face_coords = self.detect_emotion(image)
                    result_text, emoji, vis_path, hist_path = self.visualize_result(
                        image, detection_result, face_coords
                    )
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
        app = ROIOptimizedEmotionRecognizer()
        demo = app.create_interface()
        demo.launch(server_name="127.0.0.1", server_port=7860, share=False)
    except Exception as e:
        print(f"应用启动失败: {str(e)}")