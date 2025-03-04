import sys
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QScrollArea
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel

class DualCameraViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Capture Window")
        self.setGeometry(100, 100, 1600, 700)  
        
        self.camera1 = cv2.VideoCapture(0)
        self.camera2 = cv2.VideoCapture(1)
        
        self.camera1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.camera2.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.camera2.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        
        self.setup_ui()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frames)
        self.timer.start(33)  # Approximately 30 FPS
            
    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        scroll_area1 = QScrollArea()
        scroll_area1.setWidgetResizable(True)
        scroll_area2 = QScrollArea()
        scroll_area2.setWidgetResizable(True)
        
        self.camera1_label = QLabel("Camera 1 not started")
        self.camera1_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.camera2_label = QLabel("Camera 2 not started")
        self.camera2_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        scroll_area1.setWidget(self.camera1_label)
        scroll_area2.setWidget(self.camera2_label)
        
        main_layout.addWidget(scroll_area1)
        main_layout.addWidget(scroll_area2)
    
    def update_frames(self):
        ret1, frame1 = self.camera1.read()
        if ret1:
            self.display_frame(frame1, self.camera1_label)
        
        ret2, frame2 = self.camera2.read()
        if ret2:
            self.display_frame(frame2, self.camera2_label)
    
    def display_frame(self, frame, label):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_img)
        pixmap_scaled = pixmap.scaled(760, 430, Qt.AspectRatioMode.KeepAspectRatio, 
                                    Qt.TransformationMode.SmoothTransformation)
        
        label.setPixmap(pixmap_scaled)
    
    def closeEvent(self, event):
        if self.camera1.isOpened():
            self.camera1.release()
        if self.camera2.isOpened():
            self.camera2.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DualCameraViewer()
    window.show()
    sys.exit(app.exec())
