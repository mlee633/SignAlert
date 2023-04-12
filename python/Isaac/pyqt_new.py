import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # slider for Epoch
        self.lb2 = QLabel('Choose number of Epoch:',self)
        self.lb2.move(10,10) 
        self.statusBar = QStatusBar(self)
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setRange(0, 100)
        self.slider.move(10, 40)
        self.slider.valueChanged.connect(self.slider_value_changed)

        # show image function - might need to use later on
        # label = QLabel()
        # label.setPixmap(QPixmap("ASL-icon.png"))
        # self.setCentralWidget(label)

        # set icon for our tool
        self.setGeometry(500,500,700,600)
        self.setWindowIcon(QIcon("C:\\Users\\healt\\OneDrive\\문서\\GitHub\\project-1-python-team_16\\python\\Isaac\\ASL-icon.png"))

        # change name of our tool to "Sign Alert"
        self.setWindowTitle("Sign Alert")

        # status bar 
        self.statusBar = QStatusBar(self)
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("Sign Alert v0.2")

        # options box
        self.lbl = QLabel('Choose Model:',self)
        self.lbl.move(10,80)
        self.onActivated()

        cb=QComboBox(self)
        cb.addItem('AlexNet')
        cb.addItem('LeNet5')
        cb.addItem('VGG16')
        cb.move(10,100)
    
        # # radio button to choose which model
        # rbtn1 = QRadioButton('AlexNet',self)
        # rbtn1.move(10,50)
        # rbtn1.setChecked(True)
        # rbtn2 = QRadioButton('LeNet5',self)
        # rbtn2.move(10,80)
        # rbtn2.setChecked(True)
        # rbtn3 = QRadioButton('VGG16',self)
        # rbtn3.move(10,110)
        # rbtn3.setChecked(True)
        # font1 = rbtn1.font()
        # font1.setPointSize(10)
        # font1.setFamily("Gadget")
        # rbtn1.setFont(font1)
        # rbtn2.setFont(font1)
        # rbtn3.setFont(font1)
        
        # # making button function for youcannotleave
        # btn1 = QPushButton(text="Hi", parent=self)      
        # btn1.move(10,10)
        # btn1.clicked.connect(self.youcannotleave)   # .clicked connects it to our "buy" variable 

        # #quitbutton = make it exit tool
        quitbtn = QPushButton("Exit",self)
        quitbtn.move(550,550)   
        quitbtn.clicked.connect(self.close)
        quitbtn.resize(100,40)  #100 is the length, 50 is the width

        #centre calling
        self.centre()
    
    # slider define
    def slider_value_changed(self):
        value = self.slider.value()
        print("Epoch "+str(value))
    
    # adjust size 
    def onActivated(self):
        self.lbl.adjustSize()
        self.lb2.adjustSize()

    # tool centre method
    def centre(self):                                    
        qtRectangle = self.frameGeometry()
        centrePoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centrePoint)
        self.move(qtRectangle.topLeft())
    # need to change it to our favours
    def youcannotleave(self):                            
        print("fuck u")
    

print("SignAlert Tool has been open")
app = QApplication(sys.argv)

window = MyWindow()
window.show()

app.exec_()
print("SignAlert Tool has been closed")
