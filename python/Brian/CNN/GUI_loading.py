import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar
from PyQt5.QtCore import QThread, pyqtSignal
import time
from tqdm import tqdm
#from CNN_model import initParam
#from CNN_model import num_epochs

class Thread(QThread):
    _signal = pyqtSignal(int)
    def __init__(self, epoch):
        super(Thread, self).__init__()
        self.epoch = epoch

    def __del__(self):
        self.wait()

    def run(self):
        for i in range(self.epoch):
            time.sleep(0.1)
            self._signal.emit(i)

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

    def startThread(self, epochVal):
        self.thread = Thread(100)
        self.thread._signal.connect(self.updateProgress) #connects the signal in run function with the updateProgress method in the form of parameter
        self.thread.start()

    def updateProgress(self, progress):
        self.pbar.setValue(progress)

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    ex.startThread(100)
    ex.show()
    sys.exit(app.exec_())