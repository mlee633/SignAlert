import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar
from PyQt5 import QtCore
#from CNN_model import initParam
#from CNN_model import num_epochs


class MyApp(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()
        self.thread = ThreadClass()
        self.thread.progress.connect(self.updateProgress)

    def initUI(self):
        #setting up the dimensions of progress bar
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 200, 25)

        self.step = 0                         # init timer as 0

        self.setWindowTitle('QProgressBar')
        self.setGeometry(300, 300, 300, 200)
        self.show()

        #self.pbar.setValue(self.step)           # update the progress bar

    def updateProgress(self, progress):
        self.pbar.setValue(progress)
        

class ThreadClass(QtCore.QThread):
    progress = QtCore.Signal(object)

    def __init__(self, parent = None):
        super(ThreadClass, self).__init__()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    #initParam(50, 26, 0.001, 20)
    sys.exit(app.exec_())