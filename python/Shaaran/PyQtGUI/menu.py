import sys
from PyQt5.QtWidgets import*
from PyQt5.QtGui import*
import Training
class MainWindow(QMainWindow):
    def __init__(self):
        self.w = None
        super().__init__()
        self.initUI()
        self.show()
        self.centre()
        self.newWindowButton.clicked.connect(self.show_new_window)
    def initUI(self):
        self.setWindowTitle("SignAlert")
        self.setGeometry(1,1,700,400)
        self.setWindowIcon(QIcon("/Users/shaaranelango/Downloads/project-1-python-team_16/python/Shaaran/PyQtGUI/icons8-sign-language-16.png"))
        self.newWindowButton = QPushButton('new window')
        self.setCentralWidget(self.newWindowButton)
        
    def show_new_window(self, checked):
        if self.w is None:
            x = Training.Test_Train(25,26,0.001,15)
            self.w = x.progressBar
            self.w.show()
            device,train,test = x.setting_up('/Users/shaaranelango/Downloads/project-1-python-team_16/dataset/sign_mnist_train.csv','/Users/shaaranelango/Downloads/project-1-python-team_16/dataset/sign_mnist_test.csv')
            trainLoad, testLoad = x.loading_up(train,test)
            x.runModel(trainLoad,testLoad,device, 'LeNet5')

            
        
    def centre(self):
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())
