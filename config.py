import os
import yaml

# 指定  运行目录
yolo_dataset_path = "./yolo_dataset"
yaml_path = os.path.join(yolo_dataset_path, "data.yaml")

# 创建 data.yaml 文件
yaml_content = {
    "train": os.path.join(yolo_dataset_path, "images/train"),  # 训练集路径
    "val": os.path.join(yolo_dataset_path, "images/val"),      # 验证集路径
    "nc": 7,  # 类别数量
    "names": ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]  # 类别名称
}

with open(yaml_path, "w") as f:
    yaml.dump(yaml_content, f)

print(f"\n✅ `data.yaml` 配置文件已创建: {yaml_path}")
with open(yaml_path, "r") as f:
    data_yaml_content = f.read()

print("\n📜 `data.yaml` 内容:")
print(data_yaml_content)