import sys
import test

from PyQt5.QtWidgets import*
from PyQt5.QtGui import*
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.centre()
        self.show()
    def initUI(self):
        self.setWindowTitle("SignAlert")
        self.setGeometry(1,1,700,400)
        self.setWindowIcon(QIcon("/Users/shaaranelango/Downloads/project-1-python-team_16/python/Shaaran/PyQtGUI/icons8-sign-language-16.png"))
        
        
    def centre(self):
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
