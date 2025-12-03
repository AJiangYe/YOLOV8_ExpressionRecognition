import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QFileDialog, \
    QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class FacialExpressionRecognitionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Facial Expression Recognition")
        self.setGeometry(100, 100, 800, 600)

        # Load the trained YOLOv8 model
        self.model = YOLO("runs/detect/train3/weights/best.pt")

        # Create widgets
        self.upload_button = QPushButton("Upload Image", self)
        self.upload_button.clicked.connect(self.upload_image)

        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.result_label = QLabel(self)
        self.result_label.setAlignment(Qt.AlignCenter)

        self.figure_canvas = FigureCanvas(plt.Figure())

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.upload_button)
        layout.addWidget(self.image_label)
        layout.addWidget(self.result_label)
        layout.addWidget(self.figure_canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.image_path = None

    def upload_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image File", "",
                                                   "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)", options=options)
        if file_path:
            self.image_path = file_path
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(400, 400, Qt.KeepAspectRatio))
            self.predict_expression()

    def predict_expression(self):
        if not self.image_path:
            QMessageBox.warning(self, "Warning", "No image uploaded.")
            return

        # Load the image
        image = cv2.imread(self.image_path)
        if image is None:
            QMessageBox.warning(self, "Warning", "Failed to load image.")
            return

        # Convert to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Perform prediction
        results = self.model.predict(image)

        # Clear previous plot
        self.figure_canvas.figure.clear()

        # Check if results is a Results object
        if hasattr(results, 'pred'):
            # Get the prediction results
            pred = results.pred[0]
            if len(pred) > 0:
                class_ids = [int(detection[-1]) for detection in pred]
                confidences = [float(detection[-2]) for detection in pred]
                class_names = [self.model.names[class_id] for class_id in class_ids]

                # Display the result
                self.result_label.setText(f"Expressions: {class_names}\nConfidences: {confidences}")

                # Plot histogram
                plt.bar(class_names, confidences)
                plt.xlabel('Expressions')
                plt.ylabel('Confidence')
                plt.title('Expression Confidences')
                self.figure_canvas.draw()
            else:
                self.result_label.setText("No expression detected.")
        elif hasattr(results, 'results'):
            # Get the prediction results
            results_df = results.results.pandas().xyxy[0]
            if not results_df.empty:
                class_ids = results_df['class'].tolist()
                confidences = results_df['confidence'].tolist()
                class_names = [self.model.names[int(class_id)] for class_id in class_ids]

                # Display the result
                self.result_label.setText(f"Expressions: {class_names}\nConfidences: {confidences}")

                # Plot histogram
                plt.bar(class_names, confidences)
                plt.xlabel('Expressions')
                plt.ylabel('Confidence')
                plt.title('Expression Confidences')
                self.figure_canvas.draw()
            else:
                self.result_label.setText("No expression detected.")
        else:
            self.result_label.setText("No expression detected.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FacialExpressionRecognitionApp()
    window.show()
    sys.exit(app.exec_())