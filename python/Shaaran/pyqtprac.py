import sys
import test
import torch
import torchvision
from PyQt5.QtWidgets import QPushButton, QFileDialog, QMainWindow, QApplication, QAction
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("302 stujddd")
        self.fileOpenButton = QPushButton("press")
        self.setCentralWidget(self.fileOpenButton)
        self.fileOpenButton.clicked.connect(self.buttonClick)
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_action = QAction("&Your button", self)
        file_menu.addAction(file_action)
    def buttonClick(self):
        nameFile= QFileDialog.getOpenFileName(self,"Open training dataset",r"<Default dir>", "CSV (*.csv);;All Files (*)")
        print(nameFile)
        training_data = test.funct(nameFile[0])
    
        print(training_data[:, 0])
        
        #qimg = QImage(data,28,28,28,QImage.Format_Grayscale8)
        #print("pressed")
            
app = QApplication(sys.argv)
w = MainWindow()
w.resize(600,300)

w.show()
sys.exit(app.exec_())