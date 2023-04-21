import sys
import numpy as np
from PIL import Image
import cv2
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRect
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QHBoxLayout
import os
dir_path = os.path.dirname(os.path.realpath(__file__))


class CameraThread(QThread):
    change_pixmap_signal = pyqtSignal(QImage)

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
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return convert_to_Qt_format

    def stop(self):
        self._run_flag = False
        self.wait()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()  
        self.setWindowTitle('SignAlert')
        self.setWindowIcon(QIcon(dir_path+'/signalertlogo.png'))

    def init_ui(self):
        self.webcam_label = QLabel(self)
        self.webcam_label.setAlignment(Qt.AlignCenter)
        self.webcam_label.setMinimumSize(640, 480)

        self.camera_button_on = QPushButton('Camera On', self)
        self.camera_button_on.clicked.connect(self.on_camera_button_on_clicked)

        self.camera_button_off = QPushButton('Camera Off', self)
        self.camera_button_off.setEnabled(False)
        self.camera_button_off.clicked.connect(self.on_camera_button_off_clicked)

        self.take_photo_button = QPushButton('Take Photo', self)
        self.take_photo_button.setEnabled(False)
        self.take_photo_button.clicked.connect(self.on_take_photo_button_clicked)

        hbox_layout = QHBoxLayout()
        hbox_layout.addWidget(self.camera_button_on)
        hbox_layout.addWidget(self.camera_button_off)
        hbox_layout.addWidget(self.take_photo_button)

        vbox_layout = QVBoxLayout()
        vbox_layout.addWidget(self.webcam_label)
        vbox_layout.addLayout(hbox_layout)
        self.setLayout(vbox_layout)

        self.camera_thread = CameraThread()
        self.camera_thread.change_pixmap_signal.connect(self.update_image)

    def on_camera_button_on_clicked(self):
        self.camera_button_on.setEnabled(False)
        self.camera_button_off.setEnabled(True)
        self.take_photo_button.setEnabled(True)
        self.camera_thread._run_flag = True
        self.webcam_label.setText("Loading Webcam...")
        self.camera_thread.start()

    def on_camera_button_off_clicked(self):
        self.camera_button_off.setEnabled(False)
        self.camera_button_on.setEnabled(True)
        self.take_photo_button.setEnabled(False)
        self.camera_thread._run_flag = False
        self.webcam_label.clear()
        self.camera_thread.stop()
        self.camera_thread.cap()
        cv2.destroyAllWindows()
        self.destroy()

        # centre of webcam
    def update_image(self, qt_img):
        painter = QPainter(qt_img)
        painter.setPen(QPen(Qt.red, 2, Qt.SolidLine))
        
        # get the dimensions of the webcam label
        label_width = self.webcam_label.width()
        label_height = self.webcam_label.height()
        
        # set the dimensions of the rectangle
        rect_width = 230
        rect_height = 280
        
        # calculate the x and y coordinates of the rectangle
        rect_x = (label_width - rect_width) // 2
        rect_y = (label_height - rect_height) // 2
        
        painter.drawRect(QRect(rect_x, rect_y, rect_width, rect_height))
        self.webcam_label.setPixmap(QPixmap.fromImage(qt_img))

        
    def on_take_photo_button_clicked(self):
        ret, cv_img = self.camera_thread.cap.read()
        if ret:
            # Crop the image to the specified dimensions
            cropped_img = cv_img[100:380, 200:430]

            # Convert the cropped image to grayscale
            gray_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

            # Resize the grayscale image to (32,32)
            resized_img = cv2.resize(gray_img, (32, 32))

            # Convert the resized image to a Qt format image
            qt_img = self.camera_thread.convert_cv_qt(resized_img)

            # Update the label pixmap with the resized image
            self.update_image(qt_img)

            # Save the resized image
            cv2.imwrite('SignAlertPhoto.jpg', resized_img)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
