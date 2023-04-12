from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox, QPushButton, QMessageBox
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt

class MainMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SignAlert')
        self.setWindowIcon(QIcon('signalert_icon.png'))
        self.setGeometry(500, 500, 700, 600)
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

        choose_model_hbox = QHBoxLayout()
        choose_model_hbox.addWidget(choose_model_label)
        choose_model_hbox.addWidget(options_bar)

        # Training and Testing section
        train_button = QPushButton('Training')
        train_button.clicked.connect(self.open_training)
        test_button = QPushButton('Testing')
        test_button.clicked.connect(self.open_testing)
        train_test_hbox = QHBoxLayout()
        train_test_hbox.addWidget(train_button)
        train_test_hbox.addWidget(test_button)

        # Exit button
        exit_button = QPushButton('Exit')
        exit_button.clicked.connect(self.show_exit_popup)

        # Main layout
        vbox = QVBoxLayout()
        vbox.addLayout(choose_model_hbox)
        vbox.addSpacing(20)
        vbox.addLayout(train_test_hbox)
        vbox.addStretch()
        vbox.addWidget(exit_button, alignment=Qt.AlignRight)

        self.setLayout(vbox)
        self.show()

    def open_training(self):
        print('Opening Training Tab...')

    def open_testing(self):
        print('Opening Testing Tab...')

    def show_exit_popup(self):
        confirm_exit = QMessageBox()
        confirm_exit.setIcon(QMessageBox.Question)
        confirm_exit.setText("Would you like to exit?")
        confirm_exit.setWindowTitle("Exit")
        confirm_exit.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        confirm_exit.setDefaultButton(QMessageBox.No)

        if confirm_exit.exec_() == QMessageBox.Yes:
            QApplication.quit()


if __name__ == '__main__':
    app = QApplication([])
    main_menu = MainMenu()
    app.exec_()
