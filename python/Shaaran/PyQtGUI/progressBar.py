import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar
from PyQt5.QtCore import*





class PBar(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 200, 25)
        
        self.step = 0                           # init timer as 0

        self.setWindowTitle('QProgressBar')
        self.setGeometry(300, 300, 300, 200)
        
    def changeValue(self,value):
        self.pbar.setValue(value)
        QApplication.processEvents() 
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = PBar()
    sys.exit(app.exec_())
