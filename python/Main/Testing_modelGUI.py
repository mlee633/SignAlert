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
        # Checking if AlexNet, if it is then we set useAlexNet to True, this is to make sure that we do the correct processing transformations.
        if model.__class__.__name__ == "AlexNet":
            self.test = Testing.TestResults(model=model, images=self.images, useAlexNet=True)
        else:
            self.test = Testing.TestResults(model=model, images=self.images, useAlexNet=False)
        self.test.show()
    
    def ModelbuttonClick(self):
        self.nameFile= QFileDialog.getOpenFileName(self,"Open training dataset",r"<Default dir>", "PTH (*.pth);;All Files (*)")[0]
       
   

    def ImagebuttonClick(self):
        self.images= QFileDialog.getOpenFileNames(self,"Open Image Files",r"<Default dir>", "Images (*.png *.jpg *.jpeg *.bmp);;All Files (*)")[0]
       

    def open_camera_window(self):
        self.camera_window = MainWindow()
        self.camera_window.show()



    def exit_application(self):
        self.close()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_gui = TestingModelGUI()
    main_gui.show()
    sys.exit(app.exec_())
