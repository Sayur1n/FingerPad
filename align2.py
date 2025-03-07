from os import ctermid
import sys
from cv2.gapi.streaming import timestamp
import serial
import time
import cv2
import threading
import pupil_apriltags
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                             QHBoxLayout, QWidget, QPushButton, QGridLayout)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap

class SerialReaderThread(QThread):
    # Define a signal that will be emitted when new data is received
    data_received = pyqtSignal(list,float)
    
    def __init__(self, port='/dev/tty.usbmodem13201', baudrate=9600):
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self.running = True
        
    def run(self):
        try:
            # Set serial port parameters
            ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=1,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
            )
            
            print(f"Successfully opened serial port: {ser.name}")
            current_line = ""
            
            while self.running:
                if ser.in_waiting > 0:
                    # Read all available data
                    data = ser.read(ser.in_waiting)
                    # Attempt to decode the data as a string
                    text = data.decode('utf-8', errors='replace')
                    
                    # Process the received data
                    for char in text:
                        if char == '\n' or char == '\r':  # If it's a newline character
                            if current_line:  # If there's data in the current line
                                # Here you can process the complete line of data
                                try:
                                    value = [float(e) for e in current_line.rstrip(';').split(',')]
                                    # Emit the signal with the parsed values
                                    time1 = time.time()
                                    print("time1",time1)
                                    self.data_received.emit(value,time1)
                                except ValueError:
                                    print(f"Error in line: {current_line}")
                                
                                # Clear the current line for the next data
                                current_line = ""
                        else:
                            # If not a newline, add the character to the current line
                            current_line += char
                
                # Short pause to reduce CPU usage
                time.sleep(0.01)
                
        except Exception as e:
            print(f"Serial read error: {e}")
        finally:
            # Ensure the serial port is closed
            if 'ser' in locals() and ser.is_open:
                ser.close()
                print("Serial port has been closed")
    
    def stop(self):
        self.running = False
        self.wait()

class CameraHandler:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.cap_time = None
        
    def initialize(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Unable to open camera {self.camera_id}")
            return False
        
        # Set resolution to 1080p (1920x1080)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        return True
    
    def grab(self):
        if self.cap is None or not self.cap.isOpened():
            if not self.initialize():
                return False
        self.cap.grab()
        self.cap_time = time.time()

    
    def retrieve(self):
        ret, frame = self.cap.retrieve()
        if ret:
            # print(f"time_cam_ret:{self.camera_id}",time.time())
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            timestamp = self.cap_time
            return frame_rgb, timestamp
        return None
        
    def close(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()


class SynchronizedCameraGrabber:
    def __init__(self, cameras):
        self.cameras = cameras
        self.barrier = threading.Barrier(len(cameras) + 1)  # +1 for main thread
        self.frames = [None] * len(cameras)
        self.timestamps = [0.0] * len(cameras)
        self.threads = []
    
    def grab_camera(self, index):
        # 等待所有线程都准备好
        self.barrier.wait()
        # 同时执行grab
        self.cameras[index].grab()
        print(f"time_cam_grab_{index}", time.time())
        # 获取图像
        self.frames[index], self.timestamps[index]= self.cameras[index].retrieve()
    
    def synchronize_grab(self):
        self.threads = []
        for i in range(len(self.cameras)):
            thread = threading.Thread(target=self.grab_camera, args=(i,))
            thread.start()
            self.threads.append(thread)
        
        self.barrier.wait()
        
        # 等待所有线程完成
        for thread in self.threads:
            thread.join()
        
        return self.frames, self.timestamps

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dual Camera Data Acquisition System")
        self.setFixedSize(1600, 720)  # Increased minimum size
        self.detector = pupil_apriltags.Detector()
        # Define real-world coordinates of the tags (assuming 1m x 1m rectangle)
        self.real_coords = np.array([
            [0, 0, 0],    # Tag 0
            [13.55, 0, 0],    # Tag 1
            [13.55, 6.5, 0],    # Tag 2
            [0, 6.5, 0]     # Tag 3
        ], dtype='float32')
        
        # Placeholder for homography matrix
        self.homography = None

        
        # Create main layout
        main_layout = QGridLayout()
        
        # Create two camera display areas
        self.camera1_label = QLabel("Camera 1")
        self.camera1_label.setMinimumSize(640, 360)  # 16:9 aspect ratio
        self.camera1_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera1_label.setStyleSheet("border: 1px solid black;")
        
        self.camera2_label = QLabel("Camera 2")
        self.camera2_label.setMinimumSize(640, 360)  # 16:9 aspect ratio
        self.camera2_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera2_label.setStyleSheet("border: 1px solid black;")
        
        # Create data display area
        self.data_label = QLabel("Current Data: Waiting for data...")
        self.data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Create touch point display area
        self.touch_label = QLabel("Touch Point")
        self.touch_label.setMinimumSize(300, 300)
        self.touch_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.touch_label.setStyleSheet("border: 1px solid black; background-color: black;")

        
        # Create start/stop buttons
        self.start_button = QPushButton("Start")
        self.start_button.clicked.connect(self.start_acquisition)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_acquisition)
        self.stop_button.setEnabled(False)
        
        # Create close button
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        self.image_count = 0
        self.touch_time = 0
        
        # Add widgets to layout
        main_layout.addWidget(self.camera1_label, 0, 0)
        main_layout.addWidget(self.camera2_label, 0, 1)
        main_layout.addWidget(self.data_label, 1, 0, 1, 2)
        main_layout.addWidget(self.touch_label, 0, 2, 2, 1)  # Span 2 rows in column 2

        
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.close_button)
        
        main_layout.addLayout(button_layout, 2, 0, 1, 2)
        
        # Set central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
        # Initialize camera handlers
        self.camera1 = CameraHandler(0)  # First camera
        self.camera2 = CameraHandler(1)  # Second camera
        
        # Initialize serial reading thread (but do not start)
        self.serial_thread = None
        
        # # Initialize timer for synchronous capture
        self.sync_timer = QTimer()
        # self.sync_timer.timeout.connect(self.synchronize_capture)
        self.sync_grabber = SynchronizedCameraGrabber([self.camera1, self.camera2])
        open('metadata.txt', 'w').close()
        
    def start_acquisition(self):
        # Create and start the serial reading thread
        self.serial_thread = SerialReaderThread()
        self.serial_thread.data_received.connect(self.on_data_received)
        # self.serial_thread.data_received.connect(self.synchronize_capture)
        self.serial_thread.start()
        
        # Initialize cameras
        if not self.camera1.initialize():
            self.data_label.setText("Camera 1 initialization failed!")
        if not self.camera2.initialize():
            self.data_label.setText("Camera 2 initialization failed!")
        
        # Start the synchronous capture timer
        # self.sync_timer.start(33)  # Approximately 30 fps
        
        # Update button states
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
    
    def stop_acquisition(self):
        # Stop the serial reading thread
        if self.serial_thread:
            self.serial_thread.stop()
            self.serial_thread = None
        
        # Stop the synchronous capture timer
        self.sync_timer.stop()
        
        # Close cameras
        self.camera1.close()
        self.camera2.close()
        
        # Update button states
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    def synchronize_capture(self,value):
        # 使用同步抓取器获取帧
        frames, timestamps = self.sync_grabber.synchronize_grab()
        # print(timestamps[0])
        # print(timestamps[1])

        if frames[0] is not None:
            if self.image_count < 30:
                self.display_image_1(frames[0], self.camera1_label)
            else:
                self.display_image_find_touchpoint(frames[0], self.camera1_label)
        
        if frames[1] is not None:
            self.display_image_istouch(frames[1], self.camera2_label, value)
        # filename_base = f"./raw_data/lct/sync_{self.image_count:04d}"
        #
        # # 保存图像
        # cv2.imwrite(f"{filename_base}_cam1.jpg", frames[0])
        # cv2.imwrite(f"{filename_base}_cam2.jpg", frames[1])
        self.image_count += 1
        
        # 保存元数据
        with open("metadata.txt", "a") as f:
            f.write(f"{self.touch_time} {timestamps[0]} {timestamps[1]}\n")
            f.flush()
        
        


    def synchronize_capture_bak(self):
        # Grab frames from both cameras simultaneously
        self.camera1.grab()
        print(f"time_cam_grab_0",time.time())
        self.camera2.grab()
        print(f"time_cam_grab_1",time.time())
        
        # Retrieve and process the frames
        frame1 = self.camera1.retrieve()
        frame2 = self.camera2.retrieve()
        
        if frame1 is not None:
            # print("time_fuck1",time.time())
            self.display_image_1(frame1, self.camera1_label)
        
        if frame2 is not None:
            # print("time_fuck2",time.time())
            self.display_image_2(frame2, self.camera2_label)
    
    def on_data_received(self, value, touch_time):
        # Update data display
        self.touch_time = touch_time
        self.synchronize_capture(value)
        value_str = ", ".join([f"{v:.2f}" for v in value])
        self.data_label.setText(f"Current Data: {value_str}")

    def detect_apriltags(self, frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Detect tags
        tags = self.detector.detect(gray)
        
        print(len(tags))
        # Filter and sort tags (assuming we have exactly 4 tags)
        if len(tags) == 4:
            # Sort tags by ID to maintain consistent order
            tags.sort(key=lambda x: x.tag_id)
            
            # Get tag corners
            # img_points = np.array([tag.corners for tag in tags])
            # img_points = img_points.reshape(4, 2)

            # Get tag center
            centers = np.array([tag.center for tag in tags])
            centers = centers.reshape(4,2)
            
            # Calculate homography
            self.homography, _ = cv2.findHomography(centers, self.real_coords[:, :2])
            projected_points = cv2.perspectiveTransform(np.array([centers], dtype=np.float32), self.homography)
            error = np.linalg.norm(projected_points[0] - self.real_coords[:, :2], axis=1)
            mean_error = np.mean(error)
            print(f"Reprojection Error : {mean_error}")
            
            


            
            # Draw detected tags
            for tag in tags:
                corners = tag.corners
                # corners 可能是浮点数，需要转成 int
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

        
        return frame

    def detect_touchpoint(self, frame):
        # 转换到 HSV 颜色空间, 注意在retrieve 中已经完成了BRG2RGB, so just RGB2HSV is fine.
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        # 蓝色的 HSV 范围（适当调整以匹配你的蓝色）
        lower_blue = np.array([100, 150, 50])
        upper_blue = np.array([140, 255, 255])

        # 生成蓝色掩码
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        # 形态学处理（去除噪点）
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # 计算轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_quad = None
        best_center = None

        for contour in contours:
            # 计算轮廓周长
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.06 * peri, True)  # 逼近多边形

            if len(approx) == 4:  # 找到四边形
                best_quad = approx  # 记录四边形
                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])  # 计算质心 x
                    cy = int(M["m01"] / M["m00"])  # 计算质心 y
                    best_center = (cx, cy)

        if best_quad is not None:
            # 画出四边形轮廓
            cv2.drawContours(frame, [best_quad], -1, (0, 255, 0), 3)

            if best_center:
                # 画出中心点
                cv2.circle(frame, best_center, 7, (255, 0, 0), -1)
                cv2.putText(frame, "Center", (best_center[0] - 30, best_center[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        return best_center




    def map_to_real_coords(self, point):
        """Map image point to real-world coordinates"""
        if self.homography is not None:
            # Convert point to homogeneous coordinates
            point = np.array([point[0], point[1], 1])
            # Transform using homography
            real_point = np.dot(self.homography, point)
            # Convert back from homogeneous coordinates
            return real_point[:2] / real_point[2]
        return None


    def display_image_1(self, frame, label):
        # Get the dimensions of the 
        frame = self.detect_apriltags(frame)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        
        # Create a QImage from the frame data
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale the image to fit the label while maintaining aspect ratio
        pixmap = pixmap.scaled(label.width(), label.height(), 
                              Qt.AspectRatioMode.KeepAspectRatio)
        
        # Set the pixmap to the label
        label.setPixmap(pixmap)
    

    def display_image_istouch(self, frame, label, value):
        # Get the dimensions of the frame
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        if sum(value) > 0:
            cv2.putText(frame, "TOUCHED", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "UNTOUCHED", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        
        # Create a QImage from the frame data
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale the image to fit the label while maintaining aspect ratio
        pixmap = pixmap.scaled(label.width(), label.height(), 
                              Qt.AspectRatioMode.KeepAspectRatio)
        
        # Set the pixmap to the label
        label.setPixmap(pixmap)

    def display_image_find_touchpoint(self, frame, label):
        # Get the dimensions of the frame
        pts = self.detect_touchpoint(frame)
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        
        # Create a QImage from the frame data
        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Scale the image to fit the label while maintaining aspect ratio
        pixmap = pixmap.scaled(label.width(), label.height(), 
                              Qt.AspectRatioMode.KeepAspectRatio)
        
        # Set the pixmap to the label
        label.setPixmap(pixmap)

        if pts is not None:
            pts = self.map_to_real_coords(pts)
            print(pts)
            # Create black image
            touch_img = np.zeros((1800, 1360, 3), dtype=np.uint8)
            # Scale and draw point
            x = int(pts[0] * 100)
            y = int(pts[1] * 100)
            if 0 <= x < 1360 and 0 <= y < 1800:
                cv2.circle(touch_img, (x, y), 50, (255, 255, 255), -1)
            touch_img = cv2.flip(touch_img, 0)
            
            # Convert to QImage
            height, width, channel = touch_img.shape
            bytes_per_line = 3 * width
            touch_qimg = QImage(touch_img.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            touch_pixmap = QPixmap.fromImage(touch_qimg)
            touch_pixmap = touch_pixmap.scaled(self.touch_label.width(), self.touch_label.height(), 
                              Qt.AspectRatioMode.KeepAspectRatio)
            self.touch_label.setPixmap(touch_pixmap)

    def closeEvent(self, event):
        # Ensure resources are released when closing the window
        self.stop_acquisition()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
