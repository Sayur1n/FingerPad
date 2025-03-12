import cv2
import serial
import time
import threading
from collections import deque
import numpy as np
import mediapipe as mp
import pupil_apriltags
from utils import SerialCommand

# 初始化MediaPipe手部解决方案
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 创建手部检测器
# min_detection_confidence: 检测置信度阈值
# min_tracking_confidence: 跟踪置信度阈值
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

detector = pupil_apriltags.Detector()
real_coords = np.array([
    [0, 0, 0],    # Tag 0
    [13.55, 0, 0],    # Tag 1
    [13.55, 6.5, 0],    # Tag 2
    [0, 6.5, 0]     # Tag 3
], dtype='float32')
# 配置参数
CAMERA1_ID = 0          # 摄像头1的ID (1080p)
CAMERA2_ID = 1          # 摄像头2的ID (4K)
SERIAL_PORT = '/dev/tty.usbmodem113201'  # 串口设备名称，根据实际情况修改
BAUD_RATE = 115200     # 波特率，根据实际情况修改
DELAY_MS = 30           # 摄像头2延迟采集的毫秒数

# 用于存储数据的deque
imu_data_deque = deque(maxlen=100)
camera2_frame_deque = deque(maxlen=15)
stop_thread = False
i2cAddress = 0x04
cmdItr_ = 0  # 从0开始，每次调用后递增
location = 128
nBytes = 6

def read_force_data(serial_port):
    try:
        ser = serial.Serial(port=serial_port, baudrate=BAUD_RATE, timeout=1)
        ser = serial.Serial(
            port=serial_port,
            baudrate=BAUD_RATE,
            timeout=1,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE
        )

        print(f"串口 {serial_port} 打开成功")
        
        while not stop_thread:
            cmdToArduino = SerialCommand.GenerateReadCommand(i2cAddress, cmdItr_, location, nBytes)
            ser.write(cmdToArduino)
            time.sleep(0.01)

            response = ser.read(22)
            timestamp = time.time()
            # print("response",response)
            byte_data = response
            sensor_data_length = (len(byte_data) - SerialCommand.TIMESTAMP_SIZE - 4) // 2
            sensor_data = [0] * sensor_data_length
    
            for i in range(sensor_data_length):
                sensor_data[i] = (byte_data[2 * i + 4 + SerialCommand.TIMESTAMP_SIZE] << 8) + byte_data[2 * i + 5 + SerialCommand.TIMESTAMP_SIZE]
            # print(sensor_data[4])
            imu_data_deque.append((timestamp, sensor_data[4]))

    except Exception as e:
        print(f"串口读取错误: {e}")
    finally:
        ser.close()
        print("串口已关闭")


# IMU数据读取线程
# def read_imu_data(serial_port):
#     try:
#         ser = serial.Serial(port=serial_port, baudrate=BAUD_RATE, timeout=1)
#         print(f"串口 {serial_port} 打开成功")
#         current_line = ""
#
#         while not stop_thread:
#             if ser.in_waiting > 0:
#                 # Read all available data
#                 data = ser.read(ser.in_waiting)
#                 # Attempt to decode the data as a string
#                 text = data.decode('utf-8', errors='replace')
#
#                 # Process the received data
#                 for char in text:
#                     if char == '\n' or char == '\r':  # If it's a newline character
#                         if current_line:  # If there's data in the current line
#                             # Here you can process the complete line of data
#                             try:
#                                 value = [float(e) for e in current_line.rstrip(';').split(',')]
#                                 # Emit the signal with the parsed values
#                                 timestamp = time.time()
#                                 imu_data_deque.append((timestamp, value))
#                             except ValueError:
#                                 print(f"Error in line: {current_line}")
#
#                             # Clear the current line for the next data
#                             current_line = ""
#                     else:
#                         # If not a newline, add the character to the current line
#                         current_line += char
#
#                 # line = ser.readline().decode('utf-8').strip()
#             # if ser.in_waiting:
#             #     line = ser.readline().decode('utf-8').strip()
#             #     timestamp = time.time()
#             #     imu_data_deque.append((timestamp, line))
#             # time.sleep(0.001)  # 小延迟避免CPU占用过高
#     except Exception as e:
#         print(f"串口读取错误: {e}")
#     finally:
#         ser.close()
#         print("串口已关闭")

# get the closet force data
def get_closest_imu_data(target_timestamp):
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


def process_hand(frame):
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
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                h, w, c = frame.shape
                x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                pts = (index_finger_tip.x * w, index_finger_tip.y * h)
                
                # circle the index fingertip
                cv2.circle(frame, (x, y), 15, (0, 0, 255), -1)  # 红色圆圈
                
                cv2.putText(frame, f"Left Hand Index: ({x}, {y})", (x-10, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame, pts


def detect_apriltags(frame):
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect tags
        tags = detector.detect(gray)
        homography = None
        
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
            homography, _ = cv2.findHomography(centers, real_coords[:, :2])
            projected_points = cv2.perspectiveTransform(np.array([centers], dtype=np.float32), homography)
            error = np.linalg.norm(projected_points[0] - real_coords[:, :2], axis=1)
            mean_error = np.mean(error)
            print(f"Reprojection Error : {mean_error}")
            
            


            
            # Draw detected tags
            for tag in tags:
                corners = tag.corners
                # corners float to int
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

        
        return frame, homography


def map_to_real_coords(point, homography):
        """Map image point to real-world coordinates"""
        if homography is not None:
            # Convert point to homogeneous coordinates
            point = np.array([point[0], point[1], 1])
            # Transform using homography
            real_point = np.dot(homography, point)
            # Convert back from homogeneous coordinates
            return real_point[:2] / real_point[2]
        return None




def main():
    cap1 = cv2.VideoCapture(CAMERA1_ID)
    cap2 = cv2.VideoCapture(CAMERA2_ID)
    
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    
    if not cap1.isOpened():
        print("cannot open cam 1")
        return
    
    if not cap2.isOpened():
        print("cannot open cam 2")
        cap1.release()
        return
    
    # start force data thread
    force_thread = threading.Thread(target=read_force_data, args=(SERIAL_PORT,), daemon=True)
    force_thread.start()
    
    print("程序已启动。按 'q' 退出。")
    sync_times = 0
    
    try:
        while True:
            # 捕获摄像头1的帧
            if cap1.grab():
                # 记录时间戳
                cam1_timestamp = time.time()
                ret1, frame1 = cap1.retrieve()
                
                if ret1:
                    # 获取最接近摄像头1时间戳的IMU数据
                    imu_data = get_closest_imu_data(cam1_timestamp)
                    
                    # 延迟指定时间后捕获摄像头2的帧
                    time.sleep(DELAY_MS / 1000.0)
                    
                    if cap2.grab():
                        cam2_timestamp = time.time()
                        ret2, frame2 = cap2.retrieve()
                        
                        if ret2:
                            # 在摄像头1的画面上显示信息
                            if imu_data:
                                imu_timestamp, imu_value = imu_data
                                time_diff = imu_timestamp - cam1_timestamp
                                cv2.putText(frame1, f"IMU data: {imu_value}", (10, 30), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                cv2.putText(frame1, f"Time diff: {time_diff*1000:.2f}ms", (10, 60), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            
                            cv2.putText(frame1, f"Cam1 time: {cam1_timestamp:.6f}", (10, 90), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame1, f"Cam2 time: {cam2_timestamp:.6f}", (10, 120), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame1, f"Delay: {(cam2_timestamp-cam1_timestamp)*1000:.2f}ms", (10, 150), 
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            if sync_times <= 5:
                                frame2, homography = detect_apriltags(frame2)
                                if sync_times == 5:
                                    print("Detect AprilTag Finished!")
                                if homography is not None:
                                    sync_times += 1
                                continue

                            frame2, pts = process_hand(frame2)
                            # e1 = time.time()
                            # print("time gap",e1-s1)
                            if pts is not None:
                                real_touch_pts = map_to_real_coords(pts, homography)
                                print(real_touch_pts)
                            
                            cv2.imshow('Camera 1 (1080p)', frame1)
                            
                            # 调整摄像头2的图像大小以便于显示（4K图像太大）
                            # 将图像缩小到原来的1/2大小
                            frame2_resized = cv2.resize(frame2, (frame2.shape[1]//2, frame2.shape[0]//2))
                            
                            cv2.imshow('Camera 2 (4K)', frame2_resized)
                            
                            camera2_frame_deque.append((cam2_timestamp, frame2))
                
                # 检查键盘输入，按q退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    global stop_thread
                    stop_thread = True  # 设置线程停止标志
                    break
            
            # time.sleep(0.001)  # 小延迟避免CPU占用过高
    
    finally:
        # 释放资源
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        print("程序已退出。")

if __name__ == "__main__":
    main()
