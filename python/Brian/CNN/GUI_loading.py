import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar, QVBoxLayout
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
        self.action = True # for continue training
        stop_training_button = QPushButton('Stop')
        stop_training_button.setGeometry(550, 450, 20, 20)
        stop_training_button.clicked.connect(self.button_clicked)
        self.pbar.setGeometry(360, 400, 25, 25)

        self.step = 0                         # init timer as 0

        self.setWindowTitle('Training in Progress')
        self.setGeometry(300, 300, 300, 200)

        self.pbar.setValue(self.step)           # update the progress bar
        vbox = QVBoxLayout()
        vbox.addWidget(self.pbar)
        vbox.addWidget(stop_training_button)
        self.setLayout(vbox)
        self.show()

    def updateProgress(self, progress):
        self.pbar.setValue(progress)
        QApplication.processEvents()
    
    def button_clicked(self):
        self.action = False

    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    #ex.startThread(100)
    ex.show()
    sys.exit(app.exec_())