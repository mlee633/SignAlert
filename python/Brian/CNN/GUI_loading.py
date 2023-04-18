import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal
import time
#from tqdm import tqdm
#from Testing import Test_Train

class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        #setting up the dimensions of progress bar
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 200, 25)

        self.step = 0                         # init timer as 0

        self.setWindowTitle('QProgressBar')
        self.setGeometry(300, 300, 300, 200)

        self.pbar.setValue(self.step)           # update the progress bar
        self.show()

    def updateProgress(self, progress):
        self.pbar.setValue(progress)
        QApplication.processEvents()

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    #ex.startThread(100)
    ex.show()
    sys.exit(app.exec_())