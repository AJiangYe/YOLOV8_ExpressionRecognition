import os
import glob
import cv2
import torch
import shutil
import numpy as np
from tqdm import tqdm  # 用于进度条
from ultralytics import YOLO

# === 1. 确保数据集存在 ===
DATASET_PATH = "../face-expression-recognition-dataset/images"  # 替换为你的数据集路径
if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"数据集未找到: {DATASET_PATH}")

# === 2. 创建 YOLO 数据集格式 ===
YOLO_DATASET_PATH = "yolo_dataset"
os.makedirs(YOLO_DATASET_PATH, exist_ok=True)

# 创建子文件夹
for subdir in ["images/train", "images/val", "labels/train", "labels/val"]:
    os.makedirs(os.path.join(YOLO_DATASET_PATH, subdir), exist_ok=True)

# === 3. 类别映射 ===
CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_MAP = {cls: i for i, cls in enumerate(CLASSES)}

# === 4. 加载 OpenCV 人脸检测模型 ===
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def process_images(image_dir, output_image_dir, output_label_dir):
    """ 遍历图片目录，检测人脸并转换为 YOLO 格式 """
    all_images = glob.glob(os.path.join(image_dir, "*/*.jpg"))
    for image_file in tqdm(all_images, desc=f"Processing {image_dir}"):
        class_name = os.path.basename(os.path.dirname(image_file))
        if class_name not in CLASS_MAP:
            continue

        img = cv2.imread(image_file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = faces[0]  # 取第一张人脸
            img_h, img_w = img.shape[:2]

            # 归一化坐标 (转换为YOLO格式)
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            width = w / img_w
            height = h / img_h

            # 生成 YOLO 标签文件
            label_file = os.path.join(output_label_dir, os.path.basename(image_file).replace(".jpg", ".txt"))
            with open(label_file, "w") as f:
                f.write(f"{CLASS_MAP[class_name]} {x_center} {y_center} {width} {height}\n")

            # 复制图片到 YOLO 数据集目录
            output_image_path = os.path.join(output_image_dir, os.path.basename(image_file))
            shutil.copy(image_file, output_image_path)

# === 5. 处理训练集和验证集（进度条） ===
process_images(os.path.join(DATASET_PATH, "train"), os.path.join(YOLO_DATASET_PATH, "images/train"), os.path.join(YOLO_DATASET_PATH, "labels/train"))
process_images(os.path.join(DATASET_PATH, "validation"), os.path.join(YOLO_DATASET_PATH, "images/val"), os.path.join(YOLO_DATASET_PATH, "labels/val"))

print("\n✅ 数据转换完成，已转换为 YOLO 格式！")

