import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QFormLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, QCoreApplication
# from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QComboBox


# def find_available_cameras():
#     index = 0
#     arr = []
#     while True:
#         cap = cv2.VideoCapture(index)
#         if not cap.read()[0]:
#             break
#         else:
#             arr.append(index)
#         cap.release()
#         index += 1
#     return arr
#
#
# available_cameras = find_available_cameras()
# print("Available cameras are indexed at:", available_cameras)
#
#
# class VideoWindow(QWidget):
#     def __init__(self, parent=None):
#         super(VideoWindow, self).__init__(parent)
#         self.setWindowTitle("Camera Selector")
#         self.setGeometry(100, 100, 640, 480)
#
#         self.dropdown = QComboBox(self)
#         self.dropdown.addItems([str(i) for i in available_cameras])
#
#         self.start_button = QPushButton("Start Camera", self)
#         self.start_button.clicked.connect(self.start_camera)
#
#         layout = QVBoxLayout()
#         layout.addWidget(self.dropdown)
#         layout.addWidget(self.start_button)
#
#         self.setLayout(layout)
#         self.cap = None
#
#     def start_camera(self):
#         if self.cap:
#             self.cap.release()
#         camera_index = int(self.dropdown.currentText())
#         self.cap = cv2.VideoCapture(camera_index)
#         if not self.cap.isOpened():
#             print("Cannot open camera")
#
#
# app = QApplication([])
# window = VideoWindow()
# window.show()
# app.exec_()

class VideoWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Camera Feed with Capture Function")
        self.setGeometry(100, 100, 800, 480)

        # Tạo một QLabel để hiển thị hình ảnh
        self.image_label = QLabel(self)
        self.image_label.resize(640, 480)

        # Tạo các nút
        self.btn_start = QPushButton('Start Camera', self)
        self.btn_stop = QPushButton('Stop Camera', self)
        self.btn_capture = QPushButton('Capture Image', self)

        # Khi nút được nhấn
        self.btn_start.clicked.connect(self.start_camera)
        self.btn_stop.clicked.connect(self.stop_camera)
        self.btn_capture.clicked.connect(self.capture_image)

        # Tạo form nhập liệu
        self.create_form()

        # Layout chính để chứa khung hình và form
        hbox = QHBoxLayout()
        hbox.addWidget(self.image_label)
        hbox.addLayout(self.form_layout)

        # Layout dọc cho cửa sổ chính
        vbox = QVBoxLayout()
        vbox.addLayout(hbox)
        vbox.addWidget(self.btn_start)
        vbox.addWidget(self.btn_stop)
        vbox.addWidget(self.btn_capture)
        self.setLayout(vbox)

        # Thiết lập camera
        self.cap = cv2.VideoCapture(1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)

    def create_form(self):
        self.form_layout = QFormLayout()
        self.input1 = QLineEdit()
        self.input2 = QLineEdit()
        self.input3 = QLineEdit()
        self.input4 = QLineEdit()

        self.form_layout.addRow("Input 1:", self.input1)
        self.form_layout.addRow("Input 2:", self.input2)
        self.form_layout.addRow("Input 3:", self.input3)
        self.form_layout.addRow("Input 4:", self.input4)

    def start_camera(self):
        if not self.cap.isOpened():  # Kiểm tra xem camera đã mở chưa
            self.cap.open(0)  # Mở camera
        self.timer.start(20)  # Bắt đầu timer để lấy hình ảnh từ camera

    def stop_camera(self):
        self.timer.stop()

    def capture_image(self):
        ret, frame = self.cap.read()
        if ret:
            cv2.imwrite('captured_image.jpg', frame)
            print("Image captured and saved!")

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h = rgb_image.shape[0]
            w = rgb_image.shape[1]
            step = rgb_image.size // h
            q_img = QImage(rgb_image.data, w, h, step, QImage.Format_RGB888)
            self.image_label.setPixmap(QPixmap.fromImage(q_img))

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        if self.cap.isOpened():
            self.cap.release()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = VideoWindow()
    window.show()
    sys.exit(app.exec_())
