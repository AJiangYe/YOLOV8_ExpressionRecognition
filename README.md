# YOLOV8_ExpressionRecognition
基于YOLOV8开发的轻量化情绪识别系统的研究与开发
[人脸情绪识别项目使用说明书.md](https://github.com/user-attachments/files/23904973/default.md)
# 基于YOLOV8的人脸情绪识别使用说明

## 📌 项目简介 

本项目基于 **YOLOv8** 进行 **人脸表情识别**，从 **面部表情数据集** 训练一个目标检测模型，识别 **愤怒 (angry)、厌恶 (disgust)、恐惧 (fear)、高兴 (happy)、中性 (neutral)、悲伤 (sad)、惊讶 (surprise)** 七种情绪。

后改进添加用户交互多界面（GUI、Web），实现**人机交互**小项目。

------

## 🎯快速开始

由于本项目已经训练好了模型，故把快速开始文件写在开头：

打开项目文件夹后，先创建conda虚拟环境，这里我用的python=3.9环境

再运行requirement.txt文件，

`pip install requirement.txt` 

进入运行文件夹

$cd ./YOLOV8_ExpressionRecognition/working $ 

运行测试py文件

`pyton emotion_recognition_V4.py`     #打开gradio界面

`python GUI.py`     #打开GUI界面

## 📁 数据集

本项目使用 **face-expression-recognition-dataset**，该数据集已按照表情类别存放在不同的文件夹：

```bash
face-expression-recognition-dataset/
│── images/
│   ├── train/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   ├── surprise/
│   ├── validation/
│   │   ├── angry/
│   │   ├── disgust/
│   │   ├── fear/
│   │   ├── happy/
│   │   ├── neutral/
│   │   ├── sad/
│   │   ├── surprise/
```

数据集包含 **训练集 (train) 和 验证集 (validation)**。 

------

## 🚀 模型选择 

本项目采用 **YOLOv8 (You Only Look Once v8)** 进行表情检测。**为什么选择 YOLOv8?**

✅ **高效性 (Efficiency)** - 具备实时目标检测能力。

 ✅ **精准度 (Accuracy)** - 适用于小目标检测，如面部表情。 

✅ **端到端训练 (End-to-end training)** - 直接从数据集中学习，不需要额外特征工程。

------

## 🔄 数据预处理 

由于原始数据为分类格式，YOLO 需要 **目标检测格式** (bounding box 标签)。我们进行了以下转换：

Since the original dataset is in classification format, YOLO requires **object detection format** (bounding box labels). We performed the following conversions:

1. **使用 OpenCV 进行人脸检测** (Use OpenCV for face detection)
2. **生成 YOLO 格式标签** (Generate YOLO format labels)
3. **重新组织数据集结构** (Reorganize dataset structure)

转换后的数据格式如下 (The converted dataset format is as follows):

```bash
yolo_dataset/
│── images/
│   ├── train/
│   ├── val/
│── labels/
│   ├── train/
│   ├── val/
```

------

## 🎯 训练 YOLOv8

使用以下命令训练 YOLOv8: Train YOLOv8 using the following command:

```python
from ultralytics import YOLO

# 加载 YOLOv8 预训练模型 
model = YOLO("yolov8n.yaml")

# 训练模型 (Train the model)
results = model.train(
    data="./data.yaml",  # 注意改成自己的数据集路径 
    epochs=50,  # 训练轮数 
    imgsz=480,  # 图像尺寸 
    batch=16,  # 批量大小 
    device="cuda" if torch.cuda.is_available() else "cpu"
)
```

训练结果会保存在 `runs/detect/trainX/` 目录下。 

------

## 📊 训练结果可视化

### 训练损失 & mAP 曲线

```python
import matplotlib.pyplot as plt
import os

train_results_dir = "runs/detect/train3"  # 替换为你的训练目录，这里我是运行了三次，所以存在了rain3中

metrics = ["results.png", "F1_curve.png", "PR_curve.png", "confusion_matrix.png"]
plt.figure(figsize=(12, 6))

for i, metric in enumerate(metrics):
    metric_path = os.path.join(train_results_dir, metric)
    if os.path.exists(metric_path):
        img = plt.imread(metric_path)
        plt.subplot(2, 2, i + 1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(metric.replace(".png", ""))

plt.tight_layout()
plt.show()
```

------

## 🎭 运行推理 

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train3/weights/best.pt")  # 载入最优模型 
results = model.predict("/kaggle/working/test_image.jpg", save=True)
```

可视化检测结果 (Visualize detection results):

```python
import cv2
import matplotlib.pyplot as plt

output_image_path = results[0].save_dir + "/test_image.jpg"
output_image = cv2.imread(output_image_path)
output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(8, 6))
plt.imshow(output_image)
plt.axis("off")
plt.title("YOLOv8 表情检测结果 (YOLOv8 Facial Expression Detection Result)")
plt.show()
```

------

## 🔚 结论 (Conclusion)

✅ **成功训练 YOLOv8 进行人脸表情情绪识别系统的研究与开发** 

✅ **实现了数据预处理、模型训练、结果可视化和推理**

✅ **未来改进方向：使用更强的模型、增加数据增强、提高 mAP** 
