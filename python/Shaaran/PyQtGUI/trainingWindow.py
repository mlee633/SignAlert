import sys
import imageGallery
import string
from PyQt5.QtWidgets import*
from PyQt5.QtCore import*



class trainWindow(QWidget):
    
    def __init__(self):
        self.nameFile = "None Selected"
        super().__init__()
        self.label1 = QLabel(str(self.nameFile), self)
        self.sliderText = QLabel()
        self.slider = QSlider(Qt.Horizontal, self)
        self.initUI()


    def initUI(self):
        grid = QGridLayout()
        grid.addWidget(self.createUserDataset(), 0, 0)
        grid.addWidget(self.createChooseModel(), 1, 0)
        grid.addWidget(self.createParameters(), 0, 1)
        self.trainGroupBox = self.createStartTraining()
        self.trainGroupBox.setEnabled(False)
        grid.addWidget(self.trainGroupBox, 1, 1)

        self.setLayout(grid)

        self.setWindowTitle('SignAlert - Train')
        self.setGeometry(300, 300, 700, 450)
        self.show()

    def createUserDataset(self):
        groupbox = QGroupBox('Choose Training Dataset (.csv format)')
        
        self.slider.setRange(50, 90)
        self.slider.setSingleStep(2)                    # Step size
        self.slider.setTickInterval(10)                 # Tick interval
        self.slider.setTickPosition(2)                  # Tick options
        self.slider.valueChanged.connect(self.sliderChange)
        fileOpenButton = QPushButton('Select Dataset')
        self.sliderText = QLabel('Train/Validation ratio: ' + str(self.slider.value()) + '/' + str(100-self.slider.value()))
        fileOpenButton.clicked.connect(self.buttonClick)
        

        vbox = QVBoxLayout()

        vbox.addWidget(fileOpenButton)
        vbox.addWidget(self.label1)
        vbox.addStretch(2)
        vbox.addWidget(self.sliderText)
        vbox.addWidget(self.slider)
        vbox.addStretch(1)
        groupbox.setLayout(vbox)

        return groupbox

    def createChooseModel(self):
        groupbox = QGroupBox('Choose type of CNN model to use:')
        radio1 = QRadioButton('LeNet5')
        radio2 = QRadioButton('AlexNet')
        radio3 = QRadioButton('BriaNet')
        radio1.setChecked(True)

        vbox = QVBoxLayout()
        vbox.addWidget(radio1)
        vbox.addWidget(radio2)
        vbox.addWidget(radio3)
        groupbox.setLayout(vbox)
        return groupbox

    def createParameters(self):
        groupbox = QGroupBox('Change Hyperparameters')
        epochBox = QSpinBox()
        epochBox.setRange(5,20)
        epochLabel = QLabel("Number of epochs (5 - 20 range)")
        epochHbox = QHBoxLayout()
        epochHbox.addWidget(epochLabel)
        epochHbox.addWidget(epochBox)
        epochHbox.addStretch(1)

        batchLabel = QLabel("Batch Size: (30 - 128 Range)")
        batchBox = QSpinBox()
        batchBox.setRange(30,128)
        batchHbox = QHBoxLayout()
        batchHbox.addWidget(batchLabel)
        batchHbox.addWidget(batchBox)
        batchHbox.addStretch(1)

        learnLabel = QLabel("Learning Rate: (0.001 - 0.1 range)")
        learnBox = QDoubleSpinBox()
        learnBox.setRange(0.001,0.1)
        learnBox.setSingleStep(0.001)
        learnBox.setDecimals(3)
        learnHbox = QHBoxLayout()
        learnHbox.addWidget(learnLabel)
        learnHbox.addWidget(learnBox)
        learnHbox.addStretch(1)
        vbox = QVBoxLayout()
        vbox.addLayout(epochHbox)
        vbox.addLayout(batchHbox)
        vbox.addLayout(learnHbox)

        #vbox.addWidget(tristatebox)
        vbox.addStretch(1)
        groupbox.setLayout(vbox)

        return groupbox

    def createStartTraining(self):
        
        groupbox = QGroupBox('Train and view dataset')
        

        # different push buttons
        startTrainButton = QPushButton('Train Model using selected dataset')
        viewImageButton = QPushButton('View images from dataset')
        self.cb = QComboBox(self)
        self.cb.addItem(None)
        for letter in string.ascii_uppercase:
            self.cb.addItem(letter)

        hbox = QHBoxLayout()
        hbox.addWidget(viewImageButton)
        hbox.addWidget(self.cb)

        vbox = QVBoxLayout()
        vbox.addWidget(startTrainButton)
        vbox.addStretch(1)
        vbox.addLayout(hbox)
        
        groupbox.setLayout(vbox)
        viewImageButton.clicked.connect(self.openDatasetViewer)
        return groupbox
    def buttonClick(self):
        self.nameFile= QFileDialog.getOpenFileName(self,"Open training dataset",r"<Default dir>", "CSV (*.csv);;All Files (*)")[0]
        self.label1.setText("Directory: \n" + self.nameFile)
        self.label1.adjustSize()       # adjust the label size automatically
        self.trainGroupBox.setEnabled(True)
    def sliderChange(self):
        self.sliderText.setText('Train/Validation ratio: ' + str(self.slider.value()) + '/' + str(100-self.slider.value()))
        self.sliderText.adjustSize()
    def openDatasetViewer(self):
        print(self.cb.currentText())
        self.w = imageGallery.imageViewer(str(self.nameFile),(self.cb.currentText()))
        self.w.show()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = trainWindow()
    sys.exit(app.exec_())
