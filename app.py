import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from math import sqrt

# Tải mô hình YOLO của bạn (v11.pt)
model = YOLO("v11.pt")
trajectory_data = []
length = 0
data_calculate = []

# model.to('cuda:0')
# Hàm nhận diện đối tượng với YOLO (lấy bounding boxes)
def detect_objects(frame):
    results = model(frame)  # Chạy mô hình YOLO
    # Kết quả trả về là một list, truy cập vào các thông tin sau:
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Tọa độ bounding box
    confidences = results[0].boxes.conf.cpu().numpy()  # Độ tin cậy của mỗi box
    class_ids = results[0].boxes.cls.cpu().numpy()  # ID lớp đối tượng
    return boxes, confidences, class_ids


# Hàm tính góc giữa sợi dây và trục Oy (góc giữa sợi dây và trục Y)
def calculate_angle_y(ball_position, pivot_position):
    dx = ball_position[0] - pivot_position[0]  # Tính khoảng cách theo trục X
    dy = ball_position[1] - pivot_position[1]  # Tính khoảng cách theo trục Y
    angle_y = np.arctan2(dx, dy)  # Tính góc giữa sợi dây và trục Oy
    return np.degrees(angle_y)  # Trả về góc tính bằng độ


# Hàm tính chiều dài của dây
def calculate_length(ball_position, pivot_position):
    length = np.sqrt(
        (ball_position[0] - pivot_position[0]) ** 2
        + (ball_position[1] - pivot_position[1]) ** 2
    )
    return length


def calculate_angular_speed(length):
    return round(sqrt(9.8 / length), 2)


def calculate_T(length):
    return round(2 * 3.14 * sqrt(length / 9.8), 2)


def calculate_f(length):
    return round(1 / calculate_T(length), 2)


def calculate_phase(length, time):
    return round(time / calculate_T(length), 2)


# Hàm vẽ hệ trục tọa độ với đơn vị tăng dần, trục tọa độ di chuyển theo điểm gắn dây
# Hàm vẽ hệ trục tọa độ với trục Ox và Oy cắt nhau tại điểm (0, 0)
# Hàm vẽ hệ trục tọa độ với trục Ox và Oy cắt nhau tại điểm (0, 0)
def draw_coordinate_system(frame, pivot_position, max_amplitude, scale=30):
    height, width, _ = frame.shape

    # Vẽ trục X (dài hết chiều ngang của màn hình) với giá trị âm và màu đen
    # Trục X sẽ cắt trục Y tại điểm (0, 0), nghĩa là trục X sẽ đi qua pivot_position[1]
    cv2.line(
        frame, (0, pivot_position[1]), (width, pivot_position[1]), (0, 0, 0), 1
    )  # Trục X
    # Vẽ trục Y (dài hết chiều dọc của màn hình) với giá trị âm và màu đen
    cv2.line(
        frame, (pivot_position[0], 0), (pivot_position[0], height), (0, 0, 0), 1
    )  # Trục Y

    # Vẽ các vạch chia đơn vị trên trục X và Y
    for i in range(0, width, scale):  # Trục X
        # Điều chỉnh vẽ vạch chia trục X
        cv2.line(
            frame, (i, pivot_position[1] - 5), (i, pivot_position[1] + 5), (0, 0, 0), 1
        )
        # Tính toán giá trị theo trục X, bắt đầu từ pivot_position[0]
        x_value = (i - pivot_position[0]) // scale
        cv2.putText(
            frame,
            f"{x_value}",
            (i, pivot_position[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )

    for i in range(0, height, scale):  # Trục Y
        # Điều chỉnh vẽ vạch chia trục Y
        cv2.line(
            frame, (pivot_position[0] - 5, i), (pivot_position[0] + 5, i), (0, 0, 0), 1
        )
        # Tính toán giá trị theo trục Y, bắt đầu từ pivot_position[1]
        y_value = (pivot_position[1] - i) // scale
        cv2.putText(
            frame,
            f"{y_value}",
            (pivot_position[0] + 10, i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
        )


# Biến toàn cục để lưu điểm gắn dây và quỹ đạo
pivot_position = None  # Không có điểm gắn dây mặc định
max_amplitude = 0  # Biên độ lớn nhất, bắt đầu bằng 0
trajectory_points = deque(
    maxlen=15
)  # Danh sách lưu trữ các điểm quỹ đạo (2 giây, 30 FPS)
max_angle = -999  # Biến lưu góc lớn nhất, khởi tạo với giá trị rất nhỏ
paused = True  # Video sẽ tạm dừng cho đến khi chọn điểm gắn dây


# Hàm để chọn điểm gắn dây từ giao diện
def set_pivot(event, x, y, flags, param):
    global pivot_position, paused
    if event == cv2.EVENT_LBUTTONDOWN:
        pivot_position = (x, y)
        print(f"Điểm gắn dây mới: {pivot_position}")
        paused = False  # Tiếp tục video sau khi chọn điểm


# Sử dụng webcam số 2 (thay vì file video)
# cap = cv2.VideoCapture(2)  # Thay đổi nếu cần
cap = cv2.VideoCapture("images/0107.mp4")

# rtsp_url = "http://192.168.2.8:4747/video/1280x720"

# Sử dụng cv2.VideoCapture để kết nối đến RTSP stream
# cap = cv2.VideoCapture(rtsp_url)

cv2.namedWindow("Pendulum Tracking")
cv2.setMouseCallback("Pendulum Tracking", set_pivot)
#
ret, first_frame = cap.read()
if not ret:
    print("Không thể đọc video!")
    cap.release()
    cv2.destroyAllWindows()
    exit()


while paused:
    temp_frame = first_frame.copy()

    cv2.putText(
        temp_frame,
        "Click to select pivot point",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2,
    )
    cv2.imshow("Pendulum Tracking", temp_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        cap.release()
        cv2.destroyAllWindows()
        exit()

# Bắt đầu thời gian thực
start_time = time.time()
while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        # Biến để lưu vị trí quả cầu
        ball_position = None

        # Phát hiện đối tượng (quả cầu và dây)
        boxes, confidences, class_ids = detect_objects(frame)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            confidence = confidences[i]
            cls = class_ids[i]

            if confidence > 0.5:  # Lọc ra những bounding box có độ tin cậy cao
                # Giả sử quả cầu có class_id = 0, bạn có thể thay đổi theo nhu cầu
                if int(cls) == 0:
                    # Vị trí của quả cầu
                    ball_position = ((int(x1) + int(x2)) // 2, (int(y1) + int(y2)) // 2)

                    # Tính góc và chiều dài nếu đã chọn điểm gắn sợi dây
                    if pivot_position:
                        angle_y = calculate_angle_y(ball_position, pivot_position)
                        length_measure = calculate_length(ball_position, pivot_position)
                        if length == 0:
                            length = length_measure
                        # Cập nhật góc lớn nhất nếu cần
                        max_angle = max(max_angle, abs(round(angle_y)))
                        current_time = round(
                            time.time() - start_time, 2
                        )  # Tính thời gian thực (giây)
                        x_position = round(
                            (ball_position[0] - pivot_position[0]) / 30, 2
                        )  # X tính theo đơn vị
                        trajectory_data.append(
                            (current_time, x_position)
                        )  # Thêm vào danh sách
                        # data_calculate.append([
                        #     current_time,
                        #     x_position,
                        #     calculate_angular_speed(length),
                        #     max_amplitude,
                        #     calculate_phase(length, current_time),
                        #     calculate_T(length),
                        #     length]
                        # )
                        print(f"Thời gian: {current_time}s, X: {x_position}")

                        # Lưu vị trí quả cầu vào danh sách quỹ đạo
                        trajectory_points.append(ball_position)
                    else:
                        start_time = time.time()

        # Vẽ hệ trục tọa độ liên tục nếu đã chọn điểm gắn dây
        if pivot_position:
            draw_coordinate_system(frame, pivot_position, max_amplitude)

            # Làm mượt quỹ đạo bằng cách vẽ các điểm trung bình
            if len(trajectory_points) > 1:
                # Vẽ quỹ đạo con lắc bằng nét đỏ và mỏng
                for i in range(1, len(trajectory_points)):
                    cv2.line(
                        frame,
                        trajectory_points[i - 1],
                        trajectory_points[i],
                        (0, 0, 255),
                        1,
                    )  # Nét mỏng màu đỏ

            # Nếu có phát hiện quả cầu, vẽ sợi dây
            if ball_position is not None:
                cv2.line(
                    frame, pivot_position, ball_position, (0, 255, 0), 2
                )  # Vẽ sợi dây

                # Vẽ hình chiếu của con lắc lên trục X và Y
                projection_x = (
                    ball_position[0],
                    pivot_position[1],
                )  # Hình chiếu lên trục X
                projection_y = (
                    pivot_position[0],
                    ball_position[1],
                )  # Hình chiếu lên trục Y

                # Vẽ hình chiếu lên trục X và trục Y
                cv2.circle(
                    frame, projection_x, 5, (0, 0, 255), -1
                )  # Hình chiếu lên trục X
                cv2.circle(
                    frame, projection_y, 5, (255, 0, 0), -1
                )  # Hình chiếu lên trục Y

                # Hiển thị tọa độ X và Y trên trục tọa độ
                x_position_unit = (
                    ball_position[0] - pivot_position[0]
                ) / 30  # Đổi sang đơn vị theo trục X
                max_amplitude = max(max_amplitude, abs(x_position_unit))
                y_position_unit = (
                    pivot_position[1] - ball_position[1]
                ) / 30  # Đổi sang đơn vị theo trục Y
                print(f"X: {x_position_unit:.2f}, Y: {y_position_unit:.2f}")
                cv2.putText(
                    frame,
                    f"X: {x_position_unit:.2f}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Y: {y_position_unit:.2f}",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Angle: {angle_y:.2f} deg",
                    (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
                # calculate_length
                cv2.putText(
                    frame,
                    f"Length: {length:.2f}",
                    (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )
        # Hiển thị video với các kết quả
        cv2.imshow("Pendulum Tracking", frame)

        # Nhấn 'q' để thoát
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# Giải phóng tài nguyên và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
# write trajectory data to csv file
with open("trajectory_data.csv", "w") as f:
    f.write("Time,X\n")
    for time, x in trajectory_data:
        f.write(f"{time},{x}\n")
if trajectory_data:
    print(f"Góc lớn nhất: {max_angle} độ")
    # biên độ
    print(f"Biên độ lớn nhất: {max_amplitude}")

    times, x_positions = zip(*trajectory_data)  # Tách thời gian và tọa độ X

    # Vẽ đồ thị
    # plt.figure(figsize=(10, 6))
    # plt.plot(times, x_positions, marker="o", linestyle="__", color="b")
    # plt.axhline(y=0, color='black', linewidth=2)  # Vẽ trục X với đường đậm
    # plt.axvline(x=0, color='black', linewidth=2)
    # plt.title("Li độ theo thời gian")
    # plt.xlabel("Thời gian (giây)")
    # plt.ylabel("Li độ X (đơn vị)")
    # plt.grid()
    # plt.show()
    smoothed_times = savgol_filter(times, window_length=11, polyorder=3)
    smoothed_x_positions = savgol_filter(x_positions, window_length=11, polyorder=3)

    # Vẽ đồ thị
    plt.figure(figsize=(10, 6))
    plt.plot(
        times,
        x_positions,
        linestyle="-",
        color="b",
        label="Dữ liệu thô",
        marker=".",
        alpha=0.3,
    )  # Vẽ các điểm dữ liệu gốc
    plt.plot(
        smoothed_times, smoothed_x_positions, linestyle="-", color="r", label="Đồ thị"
    )
    plt.axhline(y=0, color="black", linewidth=2)  # Vẽ trục X với đường đậm
    plt.axvline(x=0, color="black", linewidth=2)
    plt.title("Li độ theo thời gian")
    plt.xlabel("Thời gian (giây)")
    plt.ylabel("Li độ X (đơn vị)")
    plt.legend()
    plt.grid()
    plt.show()

# write data calculate to csv file
with open("data_calculate.csv", "w") as f:
    f.write("Time,X,Angular Speed,Amplitude,Phase,Period,Length\n")
    for data in data_calculate:
        f.write(",".join(map(str, data)) + "\n")