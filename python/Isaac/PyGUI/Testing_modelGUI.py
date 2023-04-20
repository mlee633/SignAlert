import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QIcon

from webcam_code import CameraThread
from webcam_code import MainWindow


class TestingModelGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # Title and Icon
        self.setWindowTitle('SignAlert')
        self.setWindowIcon(QIcon('C:\\Users\\healt\\OneDrive\\문서\\GitHub\\project-1-python-team_16\\python\\Isaac\\PyGUI\\ASL-icon.png'))

        # Create the "Choose Model" button
        self.choose_model_button = QPushButton("Choose Model", self)
        self.choose_model_button.clicked.connect(self.show_model_options)

        # Create the "Choose Images" button
        self.choose_images_button = QPushButton("Choose Images", self)

        # Create the "Open Webcam" button
        self.choose_webcam_button = QPushButton("Open Webcam", self)
        self.choose_webcam_button.clicked.connect(self.open_camera_window)
    
        # Create the layout for the main window
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.choose_model_button)
        button_layout.addWidget(self.choose_images_button)
        button_layout.addWidget(self.choose_webcam_button)
        main_layout.addLayout(button_layout)

        # Create the central widget and set the layout
        central_widget = QWidget(self)
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Create the "Exit" button and add it to the status bar
        exit_button = QPushButton("Exit", self)
        exit_button.clicked.connect(self.close)
        self.statusBar().addPermanentWidget(exit_button)

    def show_model_options(self):
        # Create a new tab widget
        tab_widget = QTabWidget()

        # Set the tab widget as the central widget
        self.setCentralWidget(tab_widget)

    def open_camera_window(self):
        self.camera_window = MainWindow()
        self.camera_window.show()
    

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_gui = TestingModelGUI()
    main_gui.setGeometry(300, 300, 500, 350)
    main_gui.show()
    sys.exit(app.exec_())
