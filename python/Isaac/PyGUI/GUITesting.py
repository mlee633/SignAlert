import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGridLayout, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import cv2

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QPixmap)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        self.cap = cv2.VideoCapture(0)

    def run(self):
        while self._run_flag:
            ret, cv_img = self.cap.read()
            if ret:
                qt_img = self.convert_cv_qt(cv_img)
                self.change_pixmap_signal.emit(qt_img)

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QPixmap.fromImage(QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888))
        return convert_to_Qt_format

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Testing Model GUI")
        self.setGeometry(50, 50, 600, 400)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.grid_layout = QGridLayout()
        self.central_widget.setLayout(self.grid_layout)

        self.title_label = QLabel("Testing Model GUI")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.grid_layout.addWidget(self.title_label, 0, 0, 1, 2)

        self.open_webcam_button = QPushButton("Open Webcam", self)
        self.open_webcam_button.clicked.connect(self.open_webcam)
        self.grid_layout.addWidget(self.open_webcam_button, 1, 0)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.grid_layout.addWidget(self.image_label, 2, 0, 1, 2)

        self.model_result_label = QLabel("Model result will be displayed here.")
        self.model_result_label.setAlignment(Qt.AlignCenter)
        self.grid_layout.addWidget(self.model_result_label, 3, 0, 1, 2)

    def open_webcam(self):
        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_image)
        self.camera_thread.start()

    def update_image(self, pixmap):
        self.image_label.setPixmap(pixmap)

    def closeEvent(self, event):
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.camera_thread.stop()
            event.ignore()
        else:
            event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
