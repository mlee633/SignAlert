import sys
import test
import trainingWindow
from PyQt5.QtWidgets import*
from PyQt5.QtGui import*
class MainWindow(QMainWindow):
    def __init__(self):
        self.w = None
        super().__init__()
        self.initUI()
        self.centre()
        self.show()
    def initUI(self):
        self.setWindowTitle("SignAlert")
        self.setGeometry(1,1,700,400)
        self.setWindowIcon(QIcon("/Users/shaaranelango/Downloads/project-1-python-team_16/python/Shaaran/PyQtGUI/icons8-sign-language-16.png"))
        self.newWindowButton = QPushButton('new window')
        self.setCentralWidget(self.newWindowButton)
        self.newWindowButton.clicked.connect(self.show_new_window)
    def show_new_window(self, checked):
        if self.w is None:
            self.w = trainingWindow.trainWindow()
        self.w.show()
    def centre(self):
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
