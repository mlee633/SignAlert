from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QMessageBox, QCheckBox, QColorDialog
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, QSettings
import os
import trainingWindow
import Testing_modelGUI
dir_path = os.path.dirname(os.path.realpath(__file__))
class MainMenu(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('SignAlert')
        self.setWindowIcon(QIcon(dir_path+'/ASL-icon.png'))
        self.setGeometry(300, 300, 500, 350)
        self.center()
        self.initUI()

    def center(self):
        frame_geometry = self.frameGeometry()
        center_point = QApplication.desktop().availableGeometry().center()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())

    def initUI(self):
        # SignAlert logo
        logo_label = QLabel()
        print(dir_path)
        pixmap = QPixmap(dir_path+'/signalertlogotwo.png')
        pixmap = pixmap.scaledToWidth(600)
        logo_label.setPixmap(pixmap)
        logo_label.setAlignment(Qt.AlignCenter)

        # Introduction text
        intro_label = QLabel('"HELLO, THIS IS A SIGNALERT AND WE ARE A SIGN LANGAUGE TOOL CONVERTER"')
        intro_label.setAlignment(Qt.AlignCenter)

        # Training and Testing section
        train_button = QPushButton('Training')
        train_button.clicked.connect(self.open_training)
        test_button = QPushButton('Testing')
        test_button.clicked.connect(self.open_testing)
        train_test_hbox = QHBoxLayout()
        train_test_hbox.addWidget(train_button)
        train_test_hbox.addWidget(test_button)
        train_test_hbox.setAlignment(Qt.AlignCenter)

        # Exit button
        exit_button = QPushButton('Exit')
        exit_button.clicked.connect(self.show_exit_popup)

        # Main layout
        vbox = QVBoxLayout()
        vbox.addWidget(logo_label)
        vbox.addWidget(intro_label)
        vbox.addSpacing(20)
        vbox.addLayout(train_test_hbox)
        vbox.addStretch()
        vbox.addWidget(exit_button, alignment=Qt.AlignRight)

        self.setLayout(vbox)
        self.show()

    def open_training(self):
        self.train = trainingWindow.trainWindow()
        self.train.show()

    def open_testing(self):
        self.test = Testing_modelGUI.TestingModelGUI()
        self.test.show()

    def change_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.setStyleSheet(f"background-color: {color.name()};")


    def show_exit_popup(self):
        settings = QSettings('MyCompany', 'MyApp')
        if settings.value('confirm_exit', True, type=bool):
            confirm_exit = QMessageBox(parent=None)
            confirm_exit.setIcon(QMessageBox.Question)
            confirm_exit.setText("Would you like to exit?")
            confirm_exit.setWindowTitle("Exit")
            dont_ask_again_checkbox = QCheckBox("Don't ask again")
            confirm_exit.setCheckBox(dont_ask_again_checkbox)
            confirm_exit.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            confirm_exit.setDefaultButton(QMessageBox.No)

            # Set the geometry of the message box
            confirm_exit.setGeometry(self.geometry().center().x() - confirm_exit.width() // 2,
                                    self.geometry().center().y() - confirm_exit.height() // 2,
                                    confirm_exit.width(), confirm_exit.height())

            if confirm_exit.exec_() == QMessageBox.Yes:
                settings.setValue('confirm_exit', not dont_ask_again_checkbox.isChecked())
                QApplication.quit()

        else:
            QApplication.quit()

if __name__ == '__main__':
    app = QApplication([])
    main_menu = MainMenu()
    app.exec_()

