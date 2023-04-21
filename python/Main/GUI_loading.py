import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar, QVBoxLayout, QTextBrowser
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
        self.pbar.setGeometry(360, 400, 25, 25)
        
        #setting up the dimensions and action of button
        stop_training_button = QPushButton('Stop')
        stop_training_button.setGeometry(550, 450, 20, 20)
        stop_training_button.clicked.connect(self.button_clicked) #connects (i think like a signal) to tell that method within the parameter to be executed

        #Setting up the window containing the progress bar and textbrowser
        self.setWindowTitle('Training in Progress')
        self.setGeometry(300, 300, 300, 200)

        #setting up TextBrowser 
        self.tb = QTextBrowser()
        self.tb.setAcceptRichText(True)

        self.pbar.setValue(0)           #Set progress bar
        self.action = True # for whether to continue training or not for our models

        #Setting layout of window in the following order
        vbox = QVBoxLayout()
        vbox.addWidget(self.tb)
        vbox.addWidget(self.pbar)
        vbox.addWidget(stop_training_button)
        self.setLayout(vbox)
        self.show()

    #updates the progress bar at every event that happens
    def updateProgress(self, progress):
        self.pbar.setValue(progress)
        QApplication.processEvents() # Refreshes the GUI
    
    def button_clicked(self):
        self.action = False
        self.close()

#Testing purposes    
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    #ex.startThread(100)
    ex.show()
    sys.exit(app.exec_())