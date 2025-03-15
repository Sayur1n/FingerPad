import cv2
import serial
import time
import threading
from collections import deque
import numpy as np
import mediapipe as mp
import pupil_apriltags
from utils import SerialCommand
import os
from datetime import datetime
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QHBoxLayout, QFileDialog
from PyQt6.QtCore import QThread, pyqtSignal, Qt
from PyQt6.QtGui import QImage, QPixmap

# 初始化MediaPipe手部解决方案
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 创建手部检测器
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = pupil_apriltags.Detector()
# real_coords = np.array([
#     [0, 0, 0],    # Tag 0
#     [13.55, 0, 0],    # Tag 1
#     [13.55, 6.5, 0],    # Tag 2
#     [0, 6.5, 0]     # Tag 3
# ], dtype='float32')
# real_coords = np.array([
#     [0, 0, 0],    # Tag 0
#     [19.4, 8.3, 0],    # Tag 1
#     [0, 8.3, 0],    # Tag 2
#     [19.4, 0, 0]     # Tag 3
# ], dtype='float32')
real_coords = np.array([
    [0, 8.5, 0],    # Tag 0
    [0, 0, 0],    # Tag 1
    [19.1, 0 , 0] ,    # Tag 2
    [19.1, 8.5, 0],    # Tag 3
], dtype='float32')

# 配置参数
CAMERA1_ID = 0          # 摄像头1的ID (1080p)
CAMERA2_ID = 1          # 摄像头2的ID (4K)
SERIAL_PORT = '/dev/tty.usbmodem113201'  # 串口设备名称，根据实际情况修改
BAUD_RATE = 115200     # 波特率，根据实际情况修改
DELAY_MS = 30           # 摄像头2延迟采集的毫秒数

# 用于存储数据的deque
imu_data_deque = deque(maxlen=100)
i2cAddress = 0x04
cmdItr_ = 0  # 从0开始，每次调用后递增
location = 128
nBytes = 6


class ForceDataThread(QThread):
    def __init__(self, serial_port, baudrate):
        super().__init__()
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.running = True

    def run(self):
        try:
            ser = serial.Serial(
                port=self.serial_port,
                baudrate=self.baudrate,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )

            print(f"串口 {self.serial_port} 打开成功")
            
            while self.running:
                cmdToArduino = SerialCommand.GenerateReadCommand(i2cAddress, cmdItr_, location, nBytes)
                ser.write(cmdToArduino)
                time.sleep(0.01)

                response = ser.read(22)
                timestamp = time.time()
                byte_data = response
                sensor_data_length = (len(byte_data) - SerialCommand.TIMESTAMP_SIZE - 4) // 2
                sensor_data = [0] * sensor_data_length
        
                for i in range(sensor_data_length):
                    sensor_data[i] = (byte_data[2 * i + 4 + SerialCommand.TIMESTAMP_SIZE] << 8) + byte_data[2 * i + 5 + SerialCommand.TIMESTAMP_SIZE]
                imu_data_deque.append((timestamp, sensor_data[4]))

        except Exception as e:
            print(f"串口读取错误: {e}")
        finally:
            if 'ser' in locals():
                ser.close()
                print("串口已关闭")

    def stop(self):
        self.running = False


class CameraThread(QThread):
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, float, float, object)
    homography_ready = pyqtSignal(np.ndarray)
    calibration_status = pyqtSignal(str)
    calibration_complete = pyqtSignal()
    
    def __init__(self, camera1_id, camera2_id, width1, height1, width2, height2, delay_ms):
        super().__init__()
        self.camera1_id = camera1_id
        self.camera2_id = camera2_id
        self.width1 = width1
        self.height1 = height1
        self.width2 = width2
        self.height2 = height2
        self.delay_ms = delay_ms / 1000.0  # 转换为秒
        self.running = True
        self.capturing = False
        self.recording = False
        self.homography = None
        self.sync_times = 0
        self.needs_calibration = True
        self.current_frame1 = None
        self.current_frame2 = None
        self.current_timestamp1 = None
        self.current_timestamp2 = None
        self.current_imu_data = None

    def run(self):
        cap1 = None
        cap2 = None
        
        try:
            while self.running:
                # 等待采集开始
                while not self.capturing and self.running:
                    time.sleep(0.1)
                
                if not self.running:
                    break
                
                # 初始化摄像头
                if cap1 is None:
                    cap1 = cv2.VideoCapture(self.camera1_id)
                    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.width1)
                    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height1)
                    
                    if not cap1.isOpened():
                        print(f"无法打开摄像头 {self.camera1_id}")
                        self.capturing = False
                        continue
                
                if cap2 is None:
                    cap2 = cv2.VideoCapture(self.camera2_id)
                    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.width2)
                    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height2)
                    
                    if not cap2.isOpened():
                        print(f"无法打开摄像头 {self.camera2_id}")
                        self.capturing = False
                        continue
                
                # 校准阶段
                if self.needs_calibration:
                    self.calibration_status.emit("开始AprilTag校准...")
                    self.sync_times = 0
                    
                    # 校准
                    while self.sync_times < 5 and self.capturing:
                        if cap2.grab():
                            ret2, frame2 = cap2.retrieve()
                            if ret2:
                                frame2, homography = self.detect_apriltags(frame2)
                                if homography is not None:
                                    self.homography = homography
                                    self.sync_times += 1
                                    self.calibration_status.emit(f"AprilTag校准进度: {self.sync_times}/5")
                        time.sleep(0.05)
                    
                    if self.sync_times >= 5:
                        self.calibration_status.emit("AprilTag校准完成!")
                        self.homography_ready.emit(self.homography)
                        self.needs_calibration = False
                        self.calibration_complete.emit()
                    else:
                        self.calibration_status.emit("校准中断")
                
                # 主采集循环
                while self.capturing and self.running and not self.needs_calibration:
                    # 读取camera 1
                    ret1, frame1 = cap1.read()
                    if ret1:
                        # 记录camera 1时间戳
                        timestamp1 = time.time()
                        
                        # 等待指定延迟时间
                        time.sleep(self.delay_ms)
                        
                        # 读取camera 2
                        ret2, frame2 = cap2.read()
                        if ret2:
                            timestamp2 = time.time()
                            
                            # 获取最接近当前时间戳的力传感器数据
                            imu_data = self.get_closest_imu_data(timestamp1)
                            
                            # 保存当前帧和相关数据到类变量
                            self.current_frame1 = frame1.copy()
                            self.current_frame2 = frame2.copy()
                            self.current_timestamp1 = timestamp1
                            self.current_timestamp2 = timestamp2
                            self.current_imu_data = imu_data
                            
                            # 发送两个帧到UI
                            self.frame_ready.emit(frame1, frame2, timestamp1, timestamp2, imu_data)
                    
                    time.sleep(0.01)  # 小休眠以减少CPU使用率
        
        finally:
            # 释放资源
            if cap1 is not None:
                cap1.release()
            if cap2 is not None:
                cap2.release()
    
    def start_capture(self, recalibrate=True):
        self.needs_calibration = recalibrate
        self.capturing = True
    
    def stop_capture(self):
        self.capturing = False
        self.recording = False
    
    def stop(self):
        self.capturing = False
        self.recording = False
        self.running = False

    def detect_apriltags(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect tags
        tags = detector.detect(gray)
        homography = None
        
        # Filter and sort tags (assuming we have exactly 4 tags)
        if len(tags) == 4:
            # Sort tags by ID to maintain consistent order
            tags.sort(key=lambda x: x.tag_id)
            
            # Get tag center
            centers = np.array([tag.center for tag in tags])
            centers = centers.reshape(4, 2)
            
            # Calculate homography
            homography, _ = cv2.findHomography(centers, real_coords[:, :2])
            projected_points = cv2.perspectiveTransform(np.array([centers], dtype=np.float32), homography)
            error = np.linalg.norm(projected_points[0] - real_coords[:, :2], axis=1)
            mean_error = np.mean(error)
            print(f"Reprojection Error: {mean_error}")
            
            # Draw detected tags
            for tag in tags:
                corners = tag.corners
                corners = [(int(pt[0]), int(pt[1])) for pt in corners]
                corners_arr = np.array(corners, dtype=np.int32)
                hull = cv2.convexHull(corners_arr)
                cv2.polylines(
                    frame,
                    [hull],
                    isClosed=True,
                    color=(0, 255, 0),
                    thickness=2
                )
                c_x, c_y = int(tag.center[0]), int(tag.center[1])
                cv2.circle(frame, (c_x, c_y), 5, (255, 0, 0), -1)
                cv2.putText(frame, f"ID: {tag.tag_id}", (c_x - 10, c_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # print(type(homography))
        
        return frame, homography

    def get_closest_imu_data(self, target_timestamp):
        if not imu_data_deque:
            return None
        
        closest_data = None
        min_diff = float('inf')
        
        for data in imu_data_deque:
            timestamp, _ = data
            diff = abs(timestamp - target_timestamp)
            if diff < min_diff:
                min_diff = diff
                closest_data = data
        
        return closest_data


class CameraApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        
        # 保存相关变量
        self.is_recording = False
        self.save_dir = None
        self.frame_count = 0
        
        # 启动线程
        self.force_thread = ForceDataThread(SERIAL_PORT, BAUD_RATE)
        self.force_thread.start()
        
        self.camera_thread = CameraThread(
            CAMERA1_ID, CAMERA2_ID, 
            1920, 1080,  # camera 1 分辨率
            3840, 2160,  # camera 2 分辨率
            DELAY_MS
        )
        self.homography = None
        self.camera_thread.frame_ready.connect(self.update_frames)
        self.camera_thread.calibration_status.connect(self.update_status)
        self.camera_thread.calibration_complete.connect(self.on_calibration_complete)
        self.camera_thread.homography_ready.connect(self.update_homography)
        self.camera_thread.start()

    def initUI(self):
        self.setWindowTitle('双摄像头采集程序')
        self.setFixedSize(1200, 700)
        
        # 创建主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # 创建图像显示标签
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(640, 480)
        main_layout.addWidget(self.image_label)
        
        # 创建状态标签
        self.status_label = QLabel("就绪 - 点击'开始采集'按钮开始")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # 创建数据信息标签
        self.data_label = QLabel("等待数据...")
        self.data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(self.data_label)
        
        # 创建按钮布局
        button_layout = QHBoxLayout()
        
        # 创建开始采集按钮
        self.start_button = QPushButton("开始采集")
        self.start_button.clicked.connect(self.start_capture)
        button_layout.addWidget(self.start_button)
        
        # 创建停止采集按钮
        self.stop_button = QPushButton("停止采集")
        self.stop_button.clicked.connect(self.stop_capture)
        self.stop_button.setEnabled(False)
        button_layout.addWidget(self.stop_button)
        
        # 创建开始录制按钮
        self.record_button = QPushButton("开始录制")
        self.record_button.clicked.connect(self.start_recording)
        self.record_button.setEnabled(False)
        button_layout.addWidget(self.record_button)
        
        # 创建停止录制按钮
        self.stop_record_button = QPushButton("停止录制")
        self.stop_record_button.clicked.connect(self.stop_recording)
        self.stop_record_button.setEnabled(False)
        button_layout.addWidget(self.stop_record_button)
        
        # 创建退出按钮
        self.exit_button = QPushButton("退出")
        self.exit_button.clicked.connect(self.close)
        button_layout.addWidget(self.exit_button)
        
        main_layout.addLayout(button_layout)

    def start_capture(self):
        self.status_label.setText("正在启动摄像头...")
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.camera_thread.start_capture(recalibrate=True)
    
    def stop_capture(self):
        self.camera_thread.stop_capture()
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.record_button.setEnabled(False)
        self.stop_record_button.setEnabled(False)
        self.is_recording = False
        self.status_label.setText("采集已停止")
    
    def start_recording(self):
        # 创建保存目录
        save_dir = './raw_data/lct/'
        if not save_dir:
            return
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = os.path.join(save_dir, f"capture_{timestamp}")
        os.makedirs(self.save_dir, exist_ok=True)
        
        self.is_recording = True
        self.frame_count = 0
        self.record_button.setEnabled(False)
        self.stop_record_button.setEnabled(True)
        self.status_label.setText(f"正在录制... 保存到 {self.save_dir}")
    
    def stop_recording(self):
        self.is_recording = False
        self.record_button.setEnabled(True)
        self.stop_record_button.setEnabled(False)
        self.status_label.setText("录制已停止，继续采集中")

    def process_hand(self,frame):
        # convert from opencv brg to rgb format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 处理帧并检测手部
        results = hands.process(frame_rgb)
        pts = None
        
        # 如果检测到手部
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                is_right = handedness.classification[0].label == "Left"
                
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                
                if is_right:
                    # get index fingertip
                    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                    index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                    
                    h, w, c = frame.shape
                    x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                    pts_thumb = (thumb_tip.x * w, thumb_tip.y * h)
                    pts_index = (index_finger_tip.x * w, index_finger_tip.y * h)
                    pts_middle = (middle_finger_tip.x * w, middle_finger_tip.y *h) 
                    
                    # circle the index fingertip
                    # cv2.circle(frame, (x, y), 15, (0, 0, 255), -1)  # 红色圆圈
                    
                    # cv2.putText(frame, f"Left Hand Index: ({x}, {y})", (x-10, y-10), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    pts = [pts_thumb, pts_index, pts_middle]
        
        return frame, pts


    def map_to_real_coords(self, point, homography):
        """Map image point to real-world coordinates"""
        if homography is not None and point is not None:
            # Convert point to homogeneous coordinates
            point = np.array([point[0], point[1], 1])
            # Transform using homography
            real_point = np.dot(homography, point)
            # Convert back from homogeneous coordinates
            return real_point[:2] / real_point[2]
        return None


    
    def on_calibration_complete(self):
        self.status_label.setText("校准完成，正在采集数据...")
        self.record_button.setEnabled(True)
    
    def update_status(self, status):
        self.status_label.setText(status)

    def update_homography(self, homogrphy):
        self.homography = homogrphy

    def update_frames(self, frame1, frame2, timestamp1, timestamp2, imu_data):
        # 在主线程中保存数据
        if self.is_recording and self.save_dir:
            self.frame_count += 1
            
            # 保存camera 1图像
            
            # 保存camera 2图像
            # real_point = ['none','none']
            g1 = time.time()
            _ , pts = self.process_hand(frame2)
            g2 = time.time()
            print(g2-g1)
            if pts is not None:
                new_pts = [self.map_to_real_coords(p,self.homography) for p in pts]
            else:
                new_pts = None
            # print(new_pts)
            # if new_pts is not None:
            #     real_point = new_pts

            



            
            # 保存时间戳和力传感器数据
            if imu_data:
                imu_timestamp, imu_value = imu_data
                # if new_pts[0] is not None and new_pts[1] is not None and new_pts[2] is not None:
                if new_pts is not None:
                    print(new_pts)
                    cam1_path = os.path.join(self.save_dir, f"cam1_{self.frame_count:06d}.jpg")
                    cv2.imwrite(cam1_path, frame1)
                    cam2_path = os.path.join(self.save_dir, f"cam2_{self.frame_count:06d}.png")
                    # cv2.imwrite(cam2_path, frame2)
                    with open(os.path.join(self.save_dir, f"metadata.txt"), 'a') as f:
                        f.write(f"cam1_time:{timestamp1},cam2_time:{timestamp2},force_time:{imu_timestamp},force_value:{imu_value},coordinates:{new_pts[0],new_pts[1],new_pts[2]}\n")
        
        # 只显示camera 1的画面
        # 在帧上添加信息
        display_frame = frame1.copy()
        
        if imu_data:
            imu_timestamp, imu_value = imu_data
            time_diff1 = imu_timestamp - timestamp1
            cv2.putText(display_frame, f"IMU data: {imu_value}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Time diff: {time_diff1*1000:.2f}ms", (10, 60), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 更新数据信息标签
            self.data_label.setText(
                f"Camera 1 时间: {timestamp1:.6f} | "
                f"Camera 2 时间: {timestamp2:.6f} | "
                f"时间差: {(timestamp2-timestamp1)*1000:.2f}ms | "
                f"IMU 时间: {imu_timestamp:.6f} | "
                f"IMU 值: {imu_value}"
            )
        
        cv2.putText(display_frame, f"Cam1 time: {timestamp1:.6f}", (10, 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Cam2 time: {timestamp2:.6f}", (10, 120), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(display_frame, f"Delay: {(timestamp2-timestamp1)*1000:.2f}ms", (10, 150), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if self.is_recording:
            cv2.putText(display_frame, f"Recording... Frame: {self.frame_count}", (10, 180), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 将OpenCV帧转换为Qt图像
        height, width, channel = display_frame.shape
        bytes_per_line = 3 * width
        q_img = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888).rgbSwapped()
        
        # 调整图像大小以适应窗口
        pixmap = QPixmap.fromImage(q_img)
        self.image_label.setPixmap(pixmap.scaled(self.image_label.width(), self.image_label.height(), 
                                                Qt.AspectRatioMode.KeepAspectRatio))

    def closeEvent(self, event):
        # 停止所有线程
        self.camera_thread.stop()
        self.force_thread.stop()
        self.camera_thread.wait()
        self.force_thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication([])
    window = CameraApp()
    window.show()
    app.exec()
