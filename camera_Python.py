import cv2

# 创建VideoCapture对象，参数为设备索引号
cap = cv2.VideoCapture(0)

# 设置摄像头的分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 设置摄像头的帧率
cap.set(cv2.CAP_PROP_FPS, 30)

# 设置输出格式为MJPG
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

# 循环读取视频帧
while True:
    # 读取摄像头的图像帧
    ret, frame = cap.read()

    # 显示图像窗口
    cv2.imshow("Camera", frame)
    # 按下'q'键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头资源
cap.release()
# 关闭窗口
cv2.destroyAllWindows()
