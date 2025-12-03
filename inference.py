import matplotlib.pyplot as plt
import os

# 训练结果目录
train_results_dir = "runs/detect/train3"  # 确保是你的训练目录

# 可视化训练损失曲线
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
output_path = './output/result.png'  # 指定保存路径和文件名
plt.savefig(output_path)  # 保存图像
plt.show()
