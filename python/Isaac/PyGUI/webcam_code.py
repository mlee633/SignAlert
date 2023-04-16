import sys
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self._run_flag = True
        

    def run(self):
        cap = cv2.VideoCapture(1)

        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                qt_img = self.convert_cv_qt(cv_img)
                self.change_pixmap_signal.emit(qt_img)

        cap.release()

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return convert_to_Qt_format

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.webcam_label = QLabel(self)
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_label.setMinimumSize(640, 480)

        self.camera_button_on = QPushButton('Camera On', self)
        self.camera_button_on.clicked.connect(self.on_camera_button_on_clicked)

        self.camera_button_off = QPushButton('Camera Off', self)
        self.camera_button_off.setEnabled(False)
        self.camera_button_off.clicked.connect(self.on_camera_button_off_clicked)

        hbox_layout = QHBoxLayout()
        hbox_layout.addWidget(self.camera_button_on)
        hbox_layout.addWidget(self.camera_button_off)

        vbox_layout = QVBoxLayout()
        vbox_layout.addWidget(self.webcam_label)
        vbox_layout.addLayout(hbox_layout)
        self.setLayout(vbox_layout)

        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_image)

    def on_camera_button_on_clicked(self):
        self.camera_button_on.setEnabled(False)
        self.camera_button_off.setEnabled(True)
        self.camera_thread._run_flag = True
        self.camera_thread.start()

    def on_camera_button_off_clicked(self):
        self.camera_button_off.setEnabled(False)
        self.camera_button_on.setEnabled(True)
        self.camera_thread._run_flag = False
        self.webcam_label.clear()
        self.camera_thread.stop()

    def update_image(self, qt_img):
        self.webcam_label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.camera_thread.stop()
            event.ignore()
        else:
            event.accept()

if __name__ == '__main__':import sys
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout

class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(0)

        while self._run_flag:
            ret, cv_img = cap.read()
            if ret:
                qt_img = self.convert_cv_qt(cv_img)
                self.change_pixmap_signal.emit(qt_img)

        cap.release()

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return convert_to_Qt_format

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.webcam_label = QLabel(self)
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_label.setMinimumSize(640, 480)

        self.camera_button_on = QPushButton('Camera On', self)
        self.camera_button_on.clicked.connect(self.on_camera_button_on_clicked)

        self.camera_button_off = QPushButton('Camera Off', self)
        self.camera_button_off.setEnabled(False)
        self.camera_button_off.clicked.connect(self.on_camera_button_off_clicked)

        hbox_layout = QHBoxLayout()
        hbox_layout.addWidget(self.camera_button_on)
        hbox_layout.addWidget(self.camera_button_off)

        vbox_layout = QVBoxLayout()
        vbox_layout.addWidget(self.webcam_label)
        vbox_layout.addLayout(hbox_layout)
        self.setLayout(vbox_layout)

        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_image)

    def on_camera_button_on_clicked(self):
        self.camera_button_on.setEnabled(False)
        self.camera_button_off.setEnabled(True)
        self.camera_thread._run_flag = True
        self.webcam_label.setText("Loading Webcam...")
        self.camera_thread.start()

    def on_camera_button_off_clicked(self):
        self.camera_button_off.setEnabled(False)
        self.camera_button_on.setEnabled(True)
        self.camera_thread._run_flag = False
        self.webcam_label.clear()
        self.camera_thread.stop()

    def update_image(self, qt_img):
        self.webcam_label.setPixmap(QPixmap.fromImage(qt_img))

    def closeEvent(self, event):
        if hasattr(self, 'camera_thread') and self.camera_thread.isRunning():
            self.camera_thread.stop()
            event.ignore()
        else:
            event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
