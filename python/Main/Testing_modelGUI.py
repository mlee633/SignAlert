import sys
import torch
import Testing
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog
from PyQt5.QtGui import QIcon
from webcam_code import MainWindow
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))

class TestingModelGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # Title and Icon
        self.initUI()
        self.Back_button = QPushButton("Back")
        self.Back_button.clicked.connect(self.back_to_main) 
        self.statusBar().addWidget(self.Back_button)
        self.Exit_button = QPushButton("Exit")
        self.Exit_button.clicked.connect(self.exit_application)
        self.statusBar().addPermanentWidget(self.Exit_button)
        


    def initUI(self):
        self.setWindowTitle('SignAlert')
        self.setWindowIcon(QIcon(dir_path+'/signalertlogo.png'))
        self.setGeometry(300, 300, 600, 450)

        # Create the "Choose Model" button
        self.choose_model_button = QPushButton("Choose Trained Model", self)
        self.choose_model_button.clicked.connect(self.ModelbuttonClick)

        # Create the "Choose Images" button
        self.choose_images_button = QPushButton("Choose Images", self)
        self.choose_images_button.clicked.connect(self.ImagebuttonClick)
        # Create the "Open Webcam" button
        self.choose_webcam_button = QPushButton("Open Webcam", self)
        self.choose_webcam_button.clicked.connect(self.open_camera_window)

        # Create the "Test Selected Images" button
        self.test_images_button = QPushButton("Test Selected Images", self)
        self.test_images_button.clicked.connect(self.openTesting)

        # Create the layout for the main window
        self.main_layout = QVBoxLayout()
        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.choose_model_button)
        self.button_layout.addWidget(self.choose_images_button)
        self.main_layout.addLayout(self.button_layout)
        self.button_layout_2 = QHBoxLayout()
        self.button_layout_2.addWidget(self.choose_webcam_button)
        self.button_layout_2.addWidget(self.test_images_button)
        self.main_layout.addLayout(self.button_layout_2)

        # Create the central widget and set the layout
        central_widget = QWidget(self)
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        # Keep a reference to the tab widget
        self.tab_widget = None
    def openTesting(self):
        model = torch.load(str(self.nameFile))
        self.test = Testing.TestResults(model=model, images=self.images)
        self.test.show()
    # def show_model_options(self):
    #     # Create a new tab widget
    #     self.tab_widget = QTabWidget()

    #     # Create a tab for choosing the model file
    #     file_tab = QWidget()
    #     file_layout = QVBoxLayout()
    #     file_label = QLabel("Choose model file:")
    #     file_layout.addWidget(file_label)
    #     file_button = QPushButton("Open", self)
    #     file_button.clicked.connect(self.ModelbuttonClick)
    #     file_layout.addWidget(file_button)
    #     file_tab.setLayout(file_layout)
    #     self.tab_widget.addTab(file_tab, "Model")

        # Set the tab widget as the central widget
        self.setCentralWidget(self.tab_widget)

    def ModelbuttonClick(self):
        self.nameFile= QFileDialog.getOpenFileName(self,"Open training dataset",r"<Default dir>", "PTH (*.pth);;All Files (*)")[0]
        print(self.nameFile)
    # def show_image_options(self):
    #     # Create a new tab widget
    #     self.tab_widget = QTabWidget()

    #     # Create a tab for choosing the image files
    #     file_tab = QWidget()
    #     file_layout = QVBoxLayout()
    #     file_label = QLabel("Choose image files:")
    #     file_layout.addWidget(file_label)
    #     file_button = QPushButton("Open", self)
    #     file_button.clicked.connect(self.ImagebuttonClick)
    #     file_layout.addWidget(file_button)
    #     file_tab.setLayout(file_layout)
    #     self.tab_widget.addTab(file_tab, "Images")

    #     # Set the tab widget as the central widget
    #     self.setCentralWidget(self.tab_widget)

    def ImagebuttonClick(self):
        self.images= QFileDialog.getOpenFileNames(self,"Open Image Files",r"<Default dir>", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)")[0]
        print(self.images)

    def open_camera_window(self):
        self.camera_window = MainWindow()
        self.camera_window.show()

    def back_to_main(self):
        # Switch back to the main widget
        self.setCentralWidget(QWidget(self))

        self.tab_widget = None
        self.initUI()

    def exit_application(self):
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_gui = TestingModelGUI()
    main_gui.show()
    sys.exit(app.exec_())
