from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

class ExpressionModel:
    def __init__(self, model_path, input_size=640):
        self.model = YOLO(model_path)
        self.input_size = input_size

    def predict(self, image_path=None, video_stream=None):
        if image_path:
            # 读取图像
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.model(image_rgb)
            # 绘制检测结果
            annotated_image = results[0].plot()
            return annotated_image
        elif video_stream:
            # 读取视频帧
            results = self.model(video_stream)
            # 绘制检测结果
            annotated_frame = results[0].plot()
            return annotated_frame
        else:
            raise ValueError("Either image_path or video_stream must be provided")