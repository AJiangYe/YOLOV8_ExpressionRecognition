import sys
from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox
from PyQt5.QtCore import Qt
from model import ExpressionModel
import cv2
from PIL import Image
import numpy as np

class ExpressionGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super(ExpressionGUI, self).__init__()
        self.setWindowTitle("人脸情绪识别")
        self.setGeometry(100, 100, 800, 600)
        self.model = ExpressionModel("./runs/detect/train3/weights/best.pt")  # 确保路径正确

        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setGeometry(10, 10, 400, 300)
        self.image_label.setText("显示图像")

        self.result_label = QtWidgets.QLabel(self)
        self.result_label.setGeometry(410, 10, 380, 300)
        self.result_label.setText("识别结果")

        self.btn_load_image = QtWidgets.QPushButton("选择图片", self)
        self.btn_load_image.setGeometry(10, 320, 100, 30)
        self.btn_load_image.clicked.connect(self.load_image)

        # self.btn_start_camera = QtWidgets.QPushButton("调用摄像头", self)
        # self.btn_start_camera.setGeometry(120, 320, 100, 30)
        # self.btn_start_camera.clicked.connect(self.start_camera)

        self.capture = cv2.VideoCapture(0)
        self.timer = QtCore.QTimer(self)

    def load_image(self):
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.xpm *.jpg);;All Files (*)", options=options)
        if file_name:
            annotated_image = self.model.predict(image_path=file_name)
            self.display_image(annotated_image)

    def start_camera(self):
        if self.capture.isOpened():
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(30)  # 30 ms interval
        else:
            QMessageBox.warning(self, "错误", "无法打开摄像头")

    def update_frame(self):
        ret, frame = self.capture.read()
        if ret:
            annotated_frame = self.model.predict(video_stream=frame)
            self.display_image(annotated_frame)

    def display_image(self, image):
        if isinstance(image, str):
            image = cv2.imread(image)
        else:
            image = Image.fromarray(image)
        image = image.convert("RGBA")
        data = np.array(image)
        qim = QtGui.QImage(data.data, data.shape[1], data.shape[0], QtGui.QImage.Format_RGBA8888)
        pixmap = QtGui.QPixmap.fromImage(qim)
        self.image_label.setPixmap(pixmap)

    def closeEvent(self, event):
        self.timer.stop()
        self.capture.release()

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = ExpressionGUI()
    window.show()
    sys.exit(app.exec_())