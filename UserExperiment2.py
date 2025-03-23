import sys
import random
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel)
from PyQt6.QtGui import QPainter, QBrush, QColor, QMouseEvent
from PyQt6.QtCore import Qt, QPoint, QTimer, QDateTime, pyqtSignal

class TargetArea(QWidget):
    target_hit = pyqtSignal()
    all_completed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.target_radius = 25  # 目标点半径
        self.current_target = None
        self.target_count = 0
        self.total_targets = 10  # 总共需要触碰的目标数
        self.setMinimumSize(600, 400)
        self.setMouseTracking(True)  # 启用鼠标追踪
        self.start = False
        

    def generate_new_target(self):
        """生成新的随机目标位置"""
        margin = self.target_radius + 5
        x = random.randint(margin, self.width() - margin)
        y = random.randint(margin, self.height() - margin)
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
            painter.setBrush(QBrush(Qt.GlobalColor.red))
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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("反应速度测试")
        self.init_ui()
        self.init_timer()

    def init_ui(self):
        # 创建主控件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 布局设置
        main_layout = QVBoxLayout(central_widget)
        
        # 目标区域
        self.target_area = TargetArea()
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
        self.start_time = QDateTime.currentDateTime()
        self.timer.start(50)  # 每50ms更新一次时间
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
        if self.target_area.target_count == self.target_area.total_targets:
            self.time_label.setText(f"测试完成！总用时：{self.elapsed_time:.2f} 秒")
        self.target_area.start = False

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
        self.statusBar().showMessage(f"剩余目标：{remaining} 个")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ReactionTestApp()
    window.show()
    sys.exit(app.exec())