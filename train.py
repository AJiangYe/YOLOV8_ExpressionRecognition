import torch
from ultralytics import YOLO

# 加载 YOLOv8 预训练模型
model = YOLO("yolov8n.yaml")  # 选择 YOLOv8n 进行训练

print("\n🚀 重新开始训练 YOLOv8，请耐心等待...")
results = model.train(
    data="./yolo_dataset/data.yaml",  # **确保传递的是文件路径**
    epochs=50,   # 训练 50 轮
    imgsz=480,    # 图片大小
    batch=16,    # 批量大小
    device="cuda" if torch.cuda.is_available() else "cpu",  # 使用 GPU（如果可用）
    verbose=True
    # save_dir="./runs/train/exp"
)

print("\n✅ 训练完成！")