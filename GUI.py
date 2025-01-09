import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import time
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5 import uic
from collections import deque
from ultralytics import YOLO
from math import sqrt
from pygrabber.dshow_graph import FilterGraph
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# Load mô hình YOLO
model = None

length = 0
data_calculate = []


def get_available_cameras() :

    devices = FilterGraph().get_input_devices()

    available_cameras = {}

    for device_index, device_name in enumerate(devices):
        available_cameras[device_index] = device_name

    return available_cameras

# exit(get_available_cameras())





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





class CVApp(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi('UI.ui', self)
        self.setWindowTitle("CV2 Application with GUI")
        self.setFixedSize(982, 887)
        self.pause = False
        # ComboBox để chọn nguồn
        list_webcam = get_available_cameras()
        combo = ["Video File"]
        for i in list_webcam:
            combo.append(f"Webcam {i} : {list_webcam[i]}")

        #plt
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
  
        # this is the Navigation widget
        # it takes the Canvas widget and a parent
        self.toolbar = NavigationToolbar(self.canvas, self)




        #xàm
        self.status_bar = self.statusBar()
        self.status_bar.addWidget(QtWidgets.QLabel("Trường THPT Lộc Thái"))
        self.status_bar.addPermanentWidget(QtWidgets.QLabel("Mai Huy Bảo - 12A4 - 2024"))



        # source controller
        self.source_combo = self.findChild(QtWidgets.QComboBox,"source_combo")
        self.load_button = self.findChild(QtWidgets.QPushButton,"load_button")
        self.path_source = self.findChild(QtWidgets.QLineEdit,"path_source")
        self.pick_source_button = self.findChild(QtWidgets.QToolButton,"pick_source_button")
        self.source_combo.currentIndexChanged.connect(self.on_combobox_change)
        self.pick_source_button.setDisabled(False)
        self.path_source.setDisabled(True)
        #video controller
        self.set_pivot_button = self.findChild(QtWidgets.QPushButton,"set_pivot_button")
        self.reset_pivot_button = self.findChild(QtWidgets.QPushButton,"reset_pivot_button")
        self.input_x = self.findChild(QtWidgets.QLineEdit,"input_x")
        self.input_y = self.findChild(QtWidgets.QLineEdit,"input_y")
        self.input_x.setText("0")
        self.input_y.setText("0")
        #action controller
        self.start_button = self.findChild(QtWidgets.QPushButton,"start_button")
        self.stop_button = self.findChild(QtWidgets.QPushButton,"stop_button")
        self.analyse_button = self.findChild(QtWidgets.QPushButton,"analyse_button")
        #model group
        self.load_model_button = self.findChild(QtWidgets.QPushButton,"load_model_button")
        self.path_model_input = self.findChild(QtWidgets.QLineEdit,"path_model_input")
        self.path_model_input.setText("models/default.pt")
        self.pick_model_button_2 = self.findChild(QtWidgets.QToolButton,"pick_model_button_2")
        # source
        self.video_label = self.findChild(QtWidgets.QLabel,"video_label")
        # parameter
        self.x_parameter = self.findChild(QtWidgets.QLineEdit,"x_parameter")
        self.y_parameter = self.findChild(QtWidgets.QLineEdit,"y_parameter")
        self.angle_parameter = self.findChild(QtWidgets.QLineEdit,"angle_parameter")
        self.length_parameter = self.findChild(QtWidgets.QLineEdit,"length_parameter")
        self.amplitude_parameter = self.findChild(QtWidgets.QLineEdit,"amplitude_parameter")


        
        self.source_combo.addItems(combo)
        self.pick_source_button.clicked.connect(self.pick_source)
        self.pick_model_button_2.clicked.connect(self.pick_model)
        self.set_pivot_button.clicked.connect(self.set_pivot)
        self.reset_pivot_button.clicked.connect(self.reset_pivot)
        self.load_button.clicked.connect(self.load_source)
        self.start_button.clicked.connect(self.start_video)
        self.start_button.setEnabled(True)
        self.stop_button.clicked.connect(self.stop_video)
        self.stop_button.setEnabled(True)
        self.load_model_button.clicked.connect(self.load_model)
        self.analyse_button.clicked.connect(self.analyse_video)




        #graph
        self.graph_layout = self.findChild(QtWidgets.QVBoxLayout,"graph_layout")
        self.graph_layout.addWidget(self.toolbar)

        self.graph_layout.addWidget(self.canvas)
        # self.figure = plt.figure()
        # start yolo
        self.model = YOLO(self.path_model_input.text())



        self.current_frame = None
        #default var
        self.current_time = 0
        self.video_width = 0
        self.video_height = 0
        self.start_time = 0
        self.max_amplitude = 0
        self.max_angle = 0
        self.trajectory_points = deque(maxlen=15)
        self.trajectory_data = []
        self.length = 0
        self.data_calculate = []
        self.paused = True
        self.pivot_position = None
        # OpenCV variables
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
    def reset_pivot(self):
        self.pivot_position = (0, 0)
        self.input_x.setText("0")
        self.input_y.setText("0")
        self.x_parameter.setText("0")
        self.y_parameter.setText("0")
        self.angle_parameter.setText("0")
        self.length_parameter.setText("0")
        self.amplitude_parameter.setText("0")
        self.max_amplitude = 0
        self.length = 0
        self.trajectory_points.clear()
        self.trajectory_data.clear()
        self.data_calculate.clear()
        self.max_angle = 0
        self.update_frame(update=not self.start_button.isEnabled())
    def analyse_video(self):
        print(self.trajectory_data)
        if self.trajectory_data:
            print(f"Góc lớn nhất: {self.max_angle:.2f} độ")
            # biên độ
            print(f"Biên độ lớn nhất: {self.max_amplitude:.2f} đơn vị")

            times, x_positions = zip(*self.trajectory_data)  # Tách thời gian và tọa độ X


            smoothed_times = savgol_filter(times, window_length=11, polyorder=3)
            smoothed_x_positions = savgol_filter(x_positions, window_length=11, polyorder=3)

            # Vẽ đồ thị
            
            self.figure.clear()
            self.figure.clf()
            # graph = self.figure(figsize=(10, 6))
            ax = self.figure.add_subplot(111)
            ax.plot(
                times,
                x_positions,
                linestyle="-",
                color="b",
                label="Dữ liệu thô",
                marker=".",
                alpha=0.3,
            )  # Vẽ các điểm dữ liệu gốc
            ax.plot(
                smoothed_times, smoothed_x_positions, linestyle="-", color="r", label="Đồ thị"
            )
            ax.axhline(y=0, color="black", linewidth=2)  # Vẽ trục X với đường đậm
            ax.axvline(x=0, color="black", linewidth=2)
            ax.set_title("Li độ theo thời gian")
            ax.set_xlabel("Thời gian (giây)")
            ax.set_ylabel("Li độ X (đơn vị)")
            ax.legend()
            ax.grid()
            # ax.imshow()
            self.canvas.draw()
    def load_model(self):
        self.model = YOLO(self.path_model_input.text())
        if self.model is not None:
            self.show_message("Infomation", "Loaded PyTorch Object Detection!", QMessageBox.Information)
            self.load_model_button.setEnabled(False)
    def on_combobox_change(self, index):
        # Lấy lựa chọn từ combobox
        selected_text = self.source_combo.currentText()
        if selected_text == "Video File":
            self.path_source.setEnabled(True)
            self.pick_source_button.setEnabled(True)
        else:
            self.path_source.setDisabled(True)
            self.pick_source_button.setDisabled(True)

    def detect_objects(self,frame):
        results = self.model(frame)  # Chạy mô hình YOLO
        # Kết quả trả về là một list, truy cập vào các thông tin sau:
        boxes = results[0].boxes.xyxy.cpu().numpy()  # Tọa độ bounding box
        confidences = results[0].boxes.conf.cpu().numpy()  # Độ tin cậy của mỗi box
        class_ids = results[0].boxes.cls.cpu().numpy()  # ID lớp đối tượng
        return boxes, confidences, class_ids

    def show_message(self, title, message, icon=QMessageBox.Information):
        """Hiển thị Infomation."""
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.setIcon(icon)
        msg_box.exec_()
    def pick_source(self):
        source_path, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv)")
        if source_path:
            self.path_source.setText(source_path)
    def pick_model(self):
        model_path, _ = QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.pt)")
        if model_path:
            self.path_model_input.setText(model_path)
        
    def mousePressEvent(self, event):
        if event.button() == 1 and self.input_x != "0" and self.input_y != "0" and self.cap is not None:  # Left mouse button
            # Lấy tọa độ chuột trong video
            mouse_x, mouse_y = event.pos().x() - 300, event.pos().y() - 33
            scale_x = self.video_width / self.video_label.width()
            scale_y = self.video_height / self.video_label.height()

            # Chuyển đổi vị trí chuột trên QLabel thành vị trí chuột trên video
            video_x = int(mouse_x * scale_x)
            video_y = int(mouse_y * scale_y)
            if video_x < 0 or video_x >= self.video_width or video_y < 0 or video_y >= self.video_height:
                return # Nếu vị trí chuột nằm ngoài video, thoát sớm
            # self.pivot_position = (mouse_x, mouse_y)
            self.pivot_position = (video_x, video_y)
            self.input_x.setText(str(video_x))
            self.input_y.setText(str(video_y))
            print(f"Điểm gắn dây mới: {self.pivot_position}")
            # Thực hiện hành động khi nhấn chuột, ví dụ chọn điểm gắn dây
            self.update_frame(update=not self.start_button.isEnabled())  # Cập nhật frame mới
            print(not self.start_button.isEnabled())
            # reset các thông số
            self.x_parameter.setText("0")
            self.y_parameter.setText("0")
            self.angle_parameter.setText("0")
            self.length_parameter.setText("0")
            self.amplitude_parameter.setText("0")

            self.max_amplitude = 0
            self.length = 0
            self.update()  # Cập nhật widget để vẽ lại nếu cần
    def load_source(self):
        source_type = self.source_combo.currentText()

        if "Webcam" in source_type:
            id_webcam = int(source_type.split()[1])
            self.cap = cv2.VideoCapture(id_webcam,cv2.CAP_DSHOW)  # Sử dụng webcam
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
            
        elif source_type == "Video File":
            video_path = self.path_source.text()
            if video_path:
                self.cap = cv2.VideoCapture(video_path)
                # scale to 640x360
                
                
            else:
                self.show_message("Infomation", "You have not selected a video file!", QMessageBox.Warning)
                return  # Người dùng không chọn file, thoát sớm
        if self.cap is None or not self.cap.isOpened():
            self.show_message("Infomation", "Unable to open video source", QMessageBox.Warning)
            # print()
            self.cap = None
        else:
            # Lấy kích thước video gốc
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            new_scale_width = 640 / width
            new_scale_height = 360 / height
            scale = min(new_scale_width, new_scale_height)

            # self.video_width = int(width)
            # self.video_height = int(height)
            self.video_width = int(width * scale)
            self.video_height = int(height * scale)


            # Cập nhật kích thước của QLabel để vừa với video
            # if source_type == "Video File":
            #     self.video_label.setFixedSize(self.video_width)

            # Đọc frame đầu tiên và hiển thị ngay lập tức mà không bắt đầu chạy

            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                frame = cv2.resize(frame, (self.video_width, self.video_height),interpolation=cv2.INTER_LINEAR)
                frame = self.process_frame(frame, test=True)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                height, width, channels = frame.shape
                bytes_per_line = channels * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                self.video_label.setPixmap(QPixmap.fromImage(q_image))
            # alert to windows

            # (f"Đã tải nguồn: {source_type} với kích thước: {width}x{height}")
            self.show_message("Infomation", "Please select the pivot attachment point before starting!", QMessageBox.Information)
            self.start_button.setEnabled(True)

    
    def start_video(self):
        if self.input_x.text() == "0" and self.input_y.text() == "0":
            return self.show_message("Infomation", "Please mark the pivot attachment point!", QMessageBox.Warning)
        if self.cap is not None:
            self.timer.start(30)  # 30ms mỗi frame (~33 FPS)
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.start_time = time.time()
            self.current_time = 0
            self.trajectory_data.clear()
        else:
            self.show_message("Infomation", "Unable to open video source!", QMessageBox.Warning)
    def stop_video(self):
        if self.cap is not None:
            self.timer.stop()
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            self.start_time = 0
        else:
            self.show_message("Infomation", "Unable to open video source!", QMessageBox.Warning)
    def set_pivot(self):
        if self.input_x.text() == "0" and self.input_y.text() == "0":
            return self.show_message("Infomation", "Please select the pivot attachment point!", QMessageBox.Warning)
        if self.cap is not None:
            self.set_pivot_button.setEnabled(True)
            self.start_time = 0
            self.pivot_position = (int(self.input_x.text()), int(self.input_y.text()))
            self.trajectory_points.clear()
            self.trajectory_data.clear()
            self.data_calculate.clear()
            self.max_angle = 0
            self.update_frame(update=not self.start_button.isEnabled())
        else:
            self.show_message("Infomation", "Unable to open video source!", QMessageBox.Warning)
    def update_frame(self,update = True):
        print("Status: ", update)
        if self.cap is not None and self.cap.isOpened():
            if (update):

                ret, frame = self.cap.read()
            else:
                ret = True
                frame = self.current_frame

            if ret:
                # Xử lý khung hình (nếu cần)
                frame = cv2.resize(frame, (self.video_width, self.video_height),interpolation=cv2.INTER_LINEAR)
                processed_frame = self.process_frame(frame)

                # Chuyển đổi frame từ BGR (OpenCV) sang RGB (PyQt5)
                frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                
                height, width, channels = frame.shape
                bytes_per_line = channels * width
                q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)

                # Hiển thị hình ảnh mà không thay đổi kích thước
                self.video_label.setPixmap(QPixmap.fromImage(q_image))
            else:
                # Dừng video nếu hết khung hình
                self.stop_video()
                self.cap.release()

    def process_frame(self, frame, test=False):
        """
        Xử lý khung hình bằng OpenCV
        (Ví dụ: YOLO, vẽ quỹ đạo con lắc, ...)
        """
        self.current_frame = frame.copy()
        ball_position = None
        pivot_position = self.pivot_position
        # Phát hiện đối tượng (quả cầu và dây)
        boxes, confidences, class_ids = self.detect_objects(frame)
        # print(boxes, confidences, class_ids)
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
                        if self.length == 0:
                            self.length = length_measure
                        # Cập nhật góc lớn nhất nếu cần
                        self.max_angle = max(self.max_angle, abs(round(angle_y)))
                        self.current_time = round(
                            time.time() - self.start_time, 2
                        )  # Tính thời gian thực (giây)
                        x_position = round(
                            (ball_position[0] - pivot_position[0]) / 30, 2
                        )  # X tính theo đơn vị
                        if test == False:
                            self.trajectory_data.append(
                                (self.current_time, x_position)
                            )
                        print(f"Thời gian: {self.current_time}s, X: {x_position}")

                        # Lưu vị trí quả cầu vào danh sách quỹ đạo
                        trajectory_points.append(ball_position)
                    else:
                        self.start_time = time.time()

        # Vẽ hệ trục tọa độ liên tục nếu đã chọn điểm gắn dây
        if pivot_position:
            draw_coordinate_system(frame, pivot_position, self.max_amplitude)

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
                self.max_amplitude = max(self.max_amplitude, abs(x_position_unit))
                y_position_unit = (
                    pivot_position[1] - ball_position[1]
                ) / 30  # Đổi sang đơn vị theo trục Y
                print(f"X: {x_position_unit:.2f}, Y: {y_position_unit:.2f}")
                self.x_parameter.setText(str(f"{x_position_unit:.2f}"))
                self.y_parameter.setText(str(f"{y_position_unit:.2f}"))
                self.angle_parameter.setText(str(f"{angle_y:.2f}"))
                self.length_parameter.setText(str(f"{self.length:.2f}"))
                self.amplitude_parameter.setText(str(f"{self.max_amplitude:.2f}"))
        return frame


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_app = CVApp()
    main_app.show()
    sys.exit(app.exec_())
