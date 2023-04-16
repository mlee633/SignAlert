import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar
from PyQt5.QtCore import QBasicTimer
#from CNN_model import initParam
#from CNN_model import num_epochs


class MyApp(QWidget.QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()


    def initUI(self):
        #setting up the dimensions of progress bar
        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(30, 40, 200, 25)

        #self.btn = QPushButton('Start', self)
        #self.btn.move(40, 80)
        #self.btn.clicked.connect(self.doAction)

        self.step = 0                         # init timer as 0

        self.setWindowTitle('QProgressBar')
        self.setGeometry(300, 300, 300, 200)
        self.show()

        #self.step = epoch + 1
        self.pbar.setValue(self.step)           # update the progress bar

    #def doAction(self):
    #    if self.timer.isActive():
    #        self.timer.stop()
     #       self.btn.setText('Start')
     #   else:
     #       self.timer.start(100, self)
     #       self.btn.setText('Stop')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    #initParam(50, 26, 0.001, 20)
    sys.exit(app.exec_())