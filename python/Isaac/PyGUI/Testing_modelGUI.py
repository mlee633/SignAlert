import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtGui import QIcon

class MainGUI(QMainWindow):
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

        # Create the layout for the main window
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.choose_model_button)
        button_layout.addWidget(self.choose_images_button)
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

        # Create a widget for the LeNet option
        lenet_widget = QWidget()
        lenet_layout = QVBoxLayout()
        lenet_layout.addWidget(QLabel("Would you like to choose: LeNet"))
        lenet_test_button = QPushButton("Start Testing", lenet_widget)
        lenet_layout.addWidget(lenet_test_button)
        lenet_widget.setLayout(lenet_layout)

        # Create a widget for the AlexNet option
        alexnet_widget = QWidget()
        alexnet_layout = QVBoxLayout()
        alexnet_layout.addWidget(QLabel("Would you like to choose: AlexNet"))
        alexnet_test_button = QPushButton("Start Testing", alexnet_widget)
        alexnet_layout.addWidget(alexnet_test_button)
        alexnet_widget.setLayout(alexnet_layout)

        # Create a widget for the BriaNet option
        briannet_widget = QWidget()
        briannet_layout = QVBoxLayout()
        briannet_layout.addWidget(QLabel("Would you like to choose: BriaNet"))
        briannet_test_button = QPushButton("Start Testing", briannet_widget)
        briannet_layout.addWidget(briannet_test_button)
        briannet_widget.setLayout(briannet_layout)

        # Add the model option widgets to the tab widget
        tab_widget.addTab(lenet_widget, "LeNet")
        tab_widget.addTab(alexnet_widget, "AlexNet")
        tab_widget.addTab(briannet_widget, "BriaNet")

        # Set the tab widget as the central widget
        self.setCentralWidget(tab_widget)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_gui = MainGUI()
    main_gui.setGeometry(300, 300, 500, 350)
    main_gui.show()
    sys.exit(app.exec_())
