import sys
import random
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QLineEdit)
from PyQt6.QtGui import QPainter, QBrush, QColor, QMouseEvent
from PyQt6.QtCore import Qt, QPoint, QTimer, QDateTime, pyqtSignal
import json
import os

import numpy as np

class TargetArea(QWidget):
    target_hit = pyqtSignal()
    all_completed = pyqtSignal()

    def __init__(self, total_targets, target_radius, parent=None):
        super().__init__(parent)
        self.target_radius = target_radius  # 目标点半径
        self.current_target = None
        self.target_count = 0
        self.epoch_count = 0
        self.total_targets = total_targets  # 总共需要触碰的目标数
        self.setMinimumSize(600, 400)
        self.setMouseTracking(True)  # 启用鼠标追踪
        self.start = False
        self.target_coordinates = []
        if os.path.exists('point_coordinates.json'):
            with open('point_coordinates.json', 'r') as f:
                annotations = json.load(f)
        for annotation in annotations:
            self.target_coordinates.append(annotation['coordinates'])

    def generate_new_target(self):
        """生成新的随机目标位置"""
        margin = self.target_radius + 5
        #x = random.randint(margin, self.width() - margin)
        #y = random.randint(margin, self.height() - margin)
        x, y = self.target_coordinates[self.epoch_count][self.target_count][0], self.target_coordinates[self.epoch_count][self.target_count][1]
        self.current_target = QPoint(x, y)
        self.update()

    def reset(self):
        """重置目标状态"""
        self.current_target = None
        self.target_count = 0
        self.update()

    def paintEvent(self, event):
        """绘制目标点"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), Qt.GlobalColor.white)
        
        # 绘制当前目标
        if self.current_target:
            color = QColor(Qt.GlobalColor.blue) if self.target_count == 0 else QColor(Qt.GlobalColor.red)
            painter.setBrush(QBrush(color))
            painter.drawEllipse(self.current_target, self.target_radius, self.target_radius)

    def mouseMoveEvent(self, event):
        """检测鼠标是否进入目标区域"""
        if self.current_target and self.target_count < self.total_targets:
            distance = ((event.position().x() - self.current_target.x())**2 +
                       (event.position().y() - self.current_target.y())**2)**0.5
            if distance <= self.target_radius and self.start:
                self.target_count += 1
                self.target_hit.emit()
                if self.target_count < self.total_targets:
                    self.generate_new_target()
                else:
                    self.current_target = None
                    self.all_completed.emit()
                self.update()

class ReactionTestApp(QMainWindow):
    def __init__(self, total_targets=10, target_radius=25):
        super().__init__()
        self.setWindowTitle("反应速度测试")
        self.target_area = TargetArea(total_targets, target_radius)
        self.user_time = []
        self.mode = 'palmpad'
        self.time_data = {}
        self.init_ui()
        self.init_timer()
        # Fix window size to prevent resizing
        self.setFixedSize(650, 500)


    def init_ui(self):
        # 创建主控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 布局设置
        main_layout = QVBoxLayout(central_widget)
        
        # 用户姓名输入区域
        name_layout = QHBoxLayout()
        name_label = QLabel("用户名:", self)
        self.name_input = QLineEdit(self)
        self.name_input.setPlaceholderText("请输入您的姓名")
        name_layout.addWidget(name_label)
        name_layout.addWidget(self.name_input)
        main_layout.addLayout(name_layout)
        
        # 目标区域
        main_layout.addWidget(self.target_area)
        
        # 控制面板
        control_layout = QHBoxLayout()
        
        # 按钮
        self.start_btn = QPushButton("开始测试", self)
        self.stop_btn = QPushButton("停止测试", self)
        self.reset_btn = QPushButton("重置", self)
        
        # 时间显示
        self.time_label = QLabel("用时：0.00 秒", self)
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # 添加控件到布局
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.reset_btn)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.time_label)
        
        # 连接信号
        self.start_btn.clicked.connect(self.start_test)
        self.stop_btn.clicked.connect(self.stop_test)
        self.reset_btn.clicked.connect(self.reset_test)
        self.target_area.target_hit.connect(self.update_target_counter)
        self.target_area.all_completed.connect(self.stop_test)

        self.update_target_counter()

    def init_timer(self):
        """初始化计时器"""
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.start_time = None
        self.elapsed_time = 0

    def start_test(self):
        """开始测试"""
        self.target_area.reset()
        self.target_area.generate_new_target()
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.target_area.start = True

    def stop_test(self):
        """停止测试"""
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.target_area.start = False
        self.time_data['user'] = self.name_input.text() if self.name_input.text() else 'User'
        if self.target_area.target_count == self.target_area.total_targets:
            self.time_label.setText(f"测试完成！总用时：{self.elapsed_time:.2f} 秒")
            self.target_area.epoch_count += 1
            self.user_time.append(self.elapsed_time)
            if self.target_area.epoch_count == 10:
                print(f'平均用时：{np.mean(self.user_time):.2f}s')
                self.time_data[self.mode] = round(np.mean(self.user_time), 2)
                if self.mode == 'palmpad':
                    self.mode = 'touchpad'
                    self.target_area.epoch_count = 0
                    self.user_time.clear()
                    return 
                else:
                    json_filename = 'hitball_record_relative.json'
                    if os.path.exists(json_filename) and os.path.getsize(json_filename) > 0:
                        with open(json_filename, 'r') as read_file:
                            existing_data = json.load(read_file)
                        with open(json_filename, 'w') as f:
                            existing_data.append(self.time_data)
                            json.dump(existing_data, f)
                    else:
                        with open(json_filename, 'w') as f:
                            json.dump([self.time_data], f)
                    self.close()
                    return
            self.start_test()

    def reset_test(self):
        """重置测试"""
        self.timer.stop()
        self.elapsed_time = 0
        self.time_label.setText("用时：0.00 秒")
        self.target_area.reset()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_time(self):
        """更新时间显示"""
        current_time = QDateTime.currentDateTime()
        self.elapsed_time = self.start_time.msecsTo(current_time) / 1000
        self.time_label.setText(f"用时：{self.elapsed_time:.2f} 秒")


    def update_target_counter(self):
        """更新目标计数显示"""
        remaining = self.target_area.total_targets - self.target_area.target_count
        mode_info = f"测试模式：{self.mode}"
        
        # 确保状态栏同时显示模式和剩余目标数
        self.statusBar().showMessage(f"{mode_info} | 剩余目标：{remaining} 个")
        
        if self.target_area.target_count == 1:
            self.start_time = QDateTime.currentDateTime()
            self.timer.start(50)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ReactionTestApp()
    window.show()
    sys.exit(app.exec())