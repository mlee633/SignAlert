import sys
import test
import torch
import torchvision
import os
#import scrollArea
from PIL import Image
from PyQt5.QtWidgets import QPushButton, QWidget, QTabWidget, QMainWindow, QAction, QFileDialog, QVBoxLayout, QApplication
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI() 
    def initUI(self):
        self.setWindowTitle("302 stujddd")
        self.fileOpenButton = QPushButton("press")
        self.setCentralWidget(self.fileOpenButton)
        self.fileOpenButton.clicked.connect(self.buttonClick)
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        file_action = QAction("&Your button", self)
        file_menu.addAction(file_action)
        trainingTab = QWidget()
        testTab = QWidget()
        tabs = QTabWidget()
        tabs.addTab(trainingTab, "Training Set")
        tabs.addTab(testTab, "Testing Set")
        vbox = QVBoxLayout()
        vbox.addWidget(tabs)
        
        
    def buttonClick(self):
        nameFile= QFileDialog.getOpenFileName(self,"Open training dataset",r"<Default dir>", "CSV (*.csv);;All Files (*)")
        print(nameFile)
        training_data = test.funct(nameFile[0])
        print(training_data.ndim)
        print(training_data[:, 0])
        print(training_data[1: , 1].ndim)
        
        image_test = training_data[:, 1:]
        rows = len(training_data)
        print(f"We have {rows} rows")

        first_image = image_test[0, :]
        first_image = first_image.reshape((28, 28))
        im = Image.fromarray(first_image)
        im = im.convert("L")
       
        im.save("your_file.jpeg")
        
            
app = QApplication(sys.argv)
w = MainWindow()
w.resize(600,300)

w.show()
sys.exit(app.exec_())