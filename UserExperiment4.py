import sys
import random
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QLineEdit,
                            QInputDialog)
from PyQt6.QtGui import QPainter, QBrush, QColor
from PyQt6.QtCore import Qt, QPoint, QTimer, QDateTime, pyqtSignal
import json
import os
import numpy as np


canvas_size = (850,400)
class TargetArea(QWidget):
    target_hit = pyqtSignal()
    all_completed = pyqtSignal()

    def __init__(self, total_targets, target_radius, parent=None):
        super().__init__(parent)
        self.target_radius = target_radius
        self.current_target = None
        self.target_count = 0
        self.epoch_count = 0
        self.total_targets = total_targets
        self.setMinimumSize(canvas_size[0], canvas_size[1])
        self.target_coordinates = []
        self.input_points = []
        self.error_distances = []
        
        if os.path.exists('point_coordinates.json'):
            with open('point_coordinates.json', 'r') as f:
                annotations = json.load(f)
            for annotation in annotations:
                self.target_coordinates.append(annotation['coordinates'])

    def generate_new_target(self):
        x, y = self.target_coordinates[self.epoch_count][self.target_count][0], \
               self.target_coordinates[self.epoch_count][self.target_count][1]
        self.current_target = QPoint(x, y)
        self.update()

    def reset(self):
        self.current_target = None
        self.target_count = 0
        self.input_points = []
        self.error_distances = []
        self.update()

    def check_hit(self, x, y):
        self.input_points.append(QPoint(x, y))
        if self.current_target:
            distance = np.sqrt((x - self.current_target.x())**2 + 
                               (y - self.current_target.y())**2)
            if distance <= self.target_radius:
                self.target_count += 1
                self.target_hit.emit()
                self.error_distances.append(distance)
                if self.target_count < self.total_targets:
                    self.generate_new_target()
                else:
                    self.current_target = None
                    self.all_completed.emit()
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), Qt.GlobalColor.white)
        
        # 绘制历史输入点
        painter.setBrush(QBrush(Qt.GlobalColor.black))
        for point in self.input_points:
            painter.drawEllipse(point, 3, 3)
        
        # 绘制当前目标
        if self.current_target:
            color = QColor(Qt.GlobalColor.blue) if self.target_count == 0 else QColor(Qt.GlobalColor.red)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(self.current_target, self.target_radius, self.target_radius)

class ReactionTestApp(QMainWindow):
    def __init__(self, total_targets=10, target_radius=35):
        super().__init__()
        self.setWindowTitle("反应速度测试")
        self.target_area = TargetArea(total_targets, target_radius)
        self.user_time = []
        self.mode = 'palmpad'
        self.time_data = {}
        self.init_ui()
        self.init_timer()
        self.setFixedSize(1000, 600)

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 用户信息输入
        name_layout = QHBoxLayout()
        name_label = QLabel("用户名:")
        self.name_input = QLineEdit()
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        main_layout.addLayout(name_layout)

        # 坐标输入
        coord_layout = QHBoxLayout()
        self.x_input = QLineEdit()
        self.y_input = QLineEdit()
        self.submit_btn = QPushButton("提交坐标")
        coord_layout.addWidget(QLabel("X:"))
        coord_layout.addWidget(self.x_input)
        coord_layout.addWidget(QLabel("Y:"))
        coord_layout.addWidget(self.y_input)
        coord_layout.addWidget(self.submit_btn)
        main_layout.addLayout(coord_layout)

        # 目标区域
        main_layout.addWidget(self.target_area)

        # 控制面板
        control_layout = QHBoxLayout()
        self.start_btn = QPushButton("开始测试")
        self.stop_btn = QPushButton("停止测试")
        self.reset_btn = QPushButton("重置")
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.reset_btn)

        self.time_label = QLabel("用时：0.00 秒")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.time_label)

        # 信号连接
        self.start_btn.clicked.connect(self.start_test)
        self.stop_btn.clicked.connect(self.stop_test)
        self.reset_btn.clicked.connect(self.reset_test)
        self.submit_btn.clicked.connect(self.process_coordinate_input)
        self.target_area.target_hit.connect(self.update_target_counter)
        self.target_area.all_completed.connect(self.stop_test)

    def process_coordinate_input(self, x=0, y=0):
        try:
            #x = int(self.x_input.text())
            #y = int(self.y_input.text())
            x = float(self.x_input.text())
            y = float(self.y_input.text())
            # 若是归一化坐标
            x = round(x * canvas_size[0])
            y = round(y * canvas_size[1])
            self.target_area.check_hit(x, y)
            self.x_input.clear()
            self.y_input.clear()
        except ValueError:
            self.statusBar().showMessage("请输入有效坐标！", 3000)

    def init_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.start_time = None
        self.elapsed_time = 0

    def start_test(self):
        self.target_area.reset()
        self.target_area.generate_new_target()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_test(self):
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        
        # 计算并显示结果
        avg_error = np.mean(self.target_area.error_distances) if self.target_area.error_distances else 0
        result_text = f"总用时：{self.elapsed_time:.2f}秒，平均误差：{avg_error:.2f}像素"
        self.time_label.setText(result_text)
        print(result_text)

        self.save_and_handle_mode(avg_error)

    def save_and_handle_mode(self, avg_error):
        self.time_data['user'] = self.name_input.text() or 'Anonymous'
        self.time_data[f'{self.mode}_time'] = round(self.elapsed_time, 2)
        self.time_data[f'{self.mode}_error'] = round(avg_error, 2)

        if self.target_area.epoch_count < 9:
            self.target_area.epoch_count += 1
            self.start_test()
        else:
            if self.mode == 'palmpad':
                self.mode = 'touchpad'
                self.target_area.epoch_count = 0
                self.start_test()
            else:
                self.save_to_file()
                self.close()

    def save_to_file(self):
        data = []
        if os.path.exists('hitball_record_absolute.json'):
            with open('hitball_record_absolute.json', 'r') as f:
                data = json.load(f)
        data.append(self.time_data)
        with open('hitball_record_absolute.json', 'w') as f:
            json.dump(data, f, indent=2)

    def reset_test(self):
        self.timer.stop()
        self.elapsed_time = 0
        self.time_label.setText("用时：0.00 秒")
        self.target_area.reset()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_time(self):
        current_time = QDateTime.currentDateTime()
        self.elapsed_time = self.start_time.msecsTo(current_time) / 1000
        self.time_label.setText(f"用时：{self.elapsed_time:.2f} 秒")

    def update_target_counter(self):
        remaining = self.target_area.total_targets - self.target_area.target_count
        self.statusBar().showMessage(f"模式：{self.mode} | 剩余目标：{remaining}个")
        
        if self.target_area.target_count == 1 and not self.timer.isActive():
            self.start_time = QDateTime.currentDateTime()
            self.timer.start(50)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ReactionTestApp(total_targets=10, target_radius=35)
    window.show()

    while True:
        inputs = input("Enter x, y: ").split()
        x = float(inputs[0])
        y = float(inputs[1])
        window.process_coordinate_input(x, y)

    sys.exit(app.exec())