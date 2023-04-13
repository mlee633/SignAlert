from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QMessageBox, QCheckBox, QColorDialog
from PyQt5.QtGui import QIcon, QColor
from PyQt5.QtCore import Qt, QSettings

class MainMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SignAlert')
        self.setWindowIcon(QIcon('C:\\Users\\healt\\OneDrive\\문서\\GitHub\\project-1-python-team_16\\python\\Isaac\\PyGUI\\ASL-icon.png'))
        self.setGeometry(300, 300, 700, 450)
        self.center()
        self.initUI()

    def center(self):
        frame_geometry = self.frameGeometry()
        center_point = QApplication.desktop().availableGeometry().center()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())

    def initUI(self):
        # Choose Model section
        choose_model_label = QLabel('Choose Model:')
        choose_model_label.setAlignment(Qt.AlignLeft)
        options_bar = QComboBox()
        options_bar.addItems(['LeNet', 'AlexNet', 'BriaNet'])

        # Select button
        select_button = QPushButton('Select')
        select_button.clicked.connect(lambda: print(f"Model {options_bar.currentText()} Selected"))

        choose_model_hbox = QHBoxLayout()
        choose_model_hbox.addWidget(choose_model_label)
        choose_model_hbox.addWidget(options_bar)
        choose_model_hbox.addWidget(select_button)

        # Training and Testing section
        train_button = QPushButton('Training')
        train_button.clicked.connect(self.open_training)
        test_button = QPushButton('Testing')
        test_button.clicked.connect(self.open_testing)
        train_test_hbox = QHBoxLayout()
        train_test_hbox.addWidget(train_button)
        train_test_hbox.addWidget(test_button)

        # Appearance section
        appearance_label = QLabel('Appearance:')
        appearance_label.setAlignment(Qt.AlignLeft)
        color_button = QPushButton('Select Color')
        color_button.clicked.connect(self.change_color)
        appearance_hbox = QHBoxLayout()
        appearance_hbox.addWidget(appearance_label)
        appearance_hbox.addWidget(color_button)

        # Exit button
        exit_button = QPushButton('Exit')
        exit_button.clicked.connect(self.show_exit_popup)

        # Main layout
        vbox = QVBoxLayout()
        vbox.addLayout(choose_model_hbox)
        vbox.addSpacing(20)
        vbox.addLayout(train_test_hbox)
        vbox.addSpacing(20)
        vbox.addLayout(appearance_hbox)
        vbox.addStretch()
        vbox.addWidget(exit_button, alignment=Qt.AlignRight)

        self.setLayout(vbox)
        self.show()

    def open_training(self):
        print('Opening Training Tab...')

    def open_testing(self):
        print('Opening Testing Tab...')

    def show_exit_popup(self):
        settings = QSettings('MyCompany', 'MyApp')
        if settings.value('confirm_exit', True, type=bool):
            confirm_exit = QMessageBox(parent=self)
            confirm_exit.setIcon(QMessageBox.Question)
            confirm_exit.setText("Would you like to exit?")
            confirm_exit.setWindowTitle("Exit")
            dont_ask_again_checkbox = QCheckBox("Don't ask again")
            confirm_exit.setCheckBox(dont_ask_again_checkbox)
            confirm_exit.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            confirm_exit.setDefaultButton(QMessageBox.No)