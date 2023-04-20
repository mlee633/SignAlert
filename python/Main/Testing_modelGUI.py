import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFileDialog
from PyQt5.QtGui import QIcon
from webcam_code import MainWindow


class TestingModelGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        # Title and Icon
        self.initUI()
        self.Back_button = QPushButton("Back")
        self.Back_button.clicked.connect(self.back_to_main)
        self.statusBar().addWidget(self.Back_button)
    def initUI(self):
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
        self.main_layout = QVBoxLayout()
        self.button_layout = QHBoxLayout()
        self.button_layout.addWidget(self.choose_model_button)
        self.button_layout.addWidget(self.choose_images_button)
        self.button_layout.addWidget(self.choose_webcam_button)
        self.main_layout.addLayout(self.button_layout)

        # Create the central widget and set the layout
        central_widget = QWidget(self)
        central_widget.setLayout(self.main_layout)
        self.setCentralWidget(central_widget)

        # Create the "Back" button and add it to the status bar
        

        # Keep a reference to the tab widget
        self.tab_widget = None
        # self.setWindowTitle('SignAlert')
        # self.setWindowIcon(QIcon('C:\\Users\\healt\\OneDrive\\문서\\GitHub\\project-1-python-team_16\\python\\Isaac\\PyGUI\\ASL-icon.png'))

        # # Create the "Choose Model" button
        # self.choose_model_button = QPushButton("Choose Model", self)
        # self.choose_model_button.clicked.connect(self.show_model_options)

        # # Create the "Choose Images" button
        # self.choose_images_button = QPushButton("Choose Images", self)

        # # Create the "Open Webcam" button
        # self.choose_webcam_button = QPushButton("Open Webcam", self)
        # self.choose_webcam_button.clicked.connect(self.open_camera_window)

        # # Create the layout for the main window
        # self.main_layout = QVBoxLayout()
        # self.button_layout = QHBoxLayout()
        # self.button_layout.addWidget(self.choose_model_button)
        # self.button_layout.addWidget(self.choose_images_button)
        # self.button_layout.addWidget(self.choose_webcam_button)
        # self.main_layout.addLayout(self.button_layout)

        # # Create the central widget and set the layout
        # central_widget = QWidget(self)
        # central_widget.setLayout(self.main_layout)
        # self.setCentralWidget(central_widget)

        # # Create the "Back" button and add it to the status bar
        # Back_button = QPushButton("Back", self)
        # Back_button.clicked.connect(self.back_to_main)
        # self.statusBar().addPermanentWidget(Back_button)

        # # Keep a reference to the tab widget
        # self.tab_widget = None

    def show_model_options(self):
        # Create a new tab widget
        self.tab_widget = QTabWidget()

        # Create a tab for choosing the model file
        file_tab = QWidget()
        file_layout = QVBoxLayout()
        file_label = QLabel("Choose model file:")
        file_layout.addWidget(file_label)
        file_button = QPushButton("Open", self)
        file_button.clicked.connect(self.buttonClick)
        file_layout.addWidget(file_button)
        file_tab.setLayout(file_layout)
        self.tab_widget.addTab(file_tab, "Model")

        # Set the tab widget as the central widget
        self.setCentralWidget(self.tab_widget)

    def open_camera_window(self):
        self.camera_window = MainWindow()
        self.camera_window.show()

    def buttonClick(self):
        self.nameFile= QFileDialog.getOpenFileName(self,"Open training dataset",r"<Default dir>", "PTH (*.pth);;All Files (*)")[0]


    def back_to_main(self):
        # Switch back to the main widget
        self.setCentralWidget(QWidget(self))

        # Check if choose_model_button object exists before attempting to add it back to the layout
        # if self.choose_model_button:
        #     self.choose_model_button.setVisible(True)
        #     button_layout = self.centralWidget().layout().itemAt(0).layout()
        #     button_layout.addWidget(self.choose_model_button)

        # Clear the reference to the tab widget
        self.tab_widget = None
        self.initUI()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_gui = TestingModelGUI()
    main_gui.setGeometry(300, 300, 600, 450)
    main_gui.show()
    sys.exit(app.exec_())
