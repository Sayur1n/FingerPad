import cv2
import numpy as np
import pupil_apriltags

def main():
    # 创建 Detector，指定要检测的 Tag 家族，可以写多个
    detector = pupil_apriltags.Detector(families="tag36h11")

    # 打开摄像头 0
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("无法打开摄像头！")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("从摄像头读取帧失败！")
            break

        # 转灰度，提升检测效率
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 检测
        results = detector.detect(gray)

        # 在图像上绘制检测信息
        for r in results:
            corners = r.corners
            # corners 可能是浮点数，需要转成 int
            corners = [(int(pt[0]), int(pt[1])) for pt in corners]

            # 用多边形连线标出外框
            # cv2.polylines(
            #     frame,
            #     [cv2.convexHull(cv2.UMat([corners], dtype='int32'))],
            #     isClosed=True,
            #     color=(0, 255, 0),
            #     thickness=2
            # )
            corners_arr = np.array(corners, dtype=np.int32)
            hull = cv2.convexHull(corners_arr)

            cv2.polylines(
                frame,
                [hull],
                isClosed=True,
                color=(0, 255, 0),
                thickness=2
            )

            # 中心点
            c_x, c_y = int(r.center[0]), int(r.center[1])
            cv2.circle(frame, (c_x, c_y), 5, (0, 0, 255), -1)

            # 显示标签 ID
            cv2.putText(frame, f"ID: {r.tag_id}", (c_x - 10, c_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow("AprilTag Detection (pupil_apriltags)", frame)

        # 按 ESC 退出
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
