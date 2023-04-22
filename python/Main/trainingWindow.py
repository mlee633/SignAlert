import sys
import imageGallery
import string
import Training
import torch
import numpy as np
from PyQt5.QtWidgets import*
from PyQt5.QtCore import*



class trainWindow(QWidget):
    # Initialising the trainWindow
    def __init__(self):
        self.nameFile = "None Selected"
        super().__init__()
        self.label1 = QLabel(str(self.nameFile), self)
        self.sliderText = QLabel()
        self.slider = QSlider(Qt.Horizontal, self)
        self.initUI()


    def initUI(self):
        # Creating a grid layout to display our information
        grid = QGridLayout()
        grid.addWidget(self.createUserDataset(), 0, 0)
        grid.addWidget(self.createChooseModel(), 1, 0)
        grid.addWidget(self.createParameters(), 0, 1)
        self.trainGroupBox = self.createStartTraining()

        # Disabling the trainingGroupBox only to enable it once a dataset has been chosen
        self.trainGroupBox.setEnabled(False)
        grid.addWidget(self.trainGroupBox, 1, 1)

        self.setLayout(grid)

        self.setWindowTitle('SignAlert - Train')
        self.setGeometry(300, 300, 1100,750)
        #self.show()

    def createUserDataset(self):

        groupbox = QGroupBox('Choose Training Dataset (.csv format)')
        
        # Train/Validation ratio set to have Train range from 50 to 90 %
        self.slider.setRange(50, 90)
        self.slider.setSingleStep(2)                    # Step size
        self.slider.setTickInterval(10)                 # Tick interval
        self.slider.setTickPosition(2)                  # Tick options

        # When the value is changed, we display the new ratio (code found in sliderChange())
        self.slider.valueChanged.connect(self.sliderChange)
        fileOpenButton = QPushButton('Select Dataset')
        self.sliderText = QLabel('Train/Validation ratio: ' + str(self.slider.value()) + '/' + str(100-self.slider.value()))
        fileOpenButton.clicked.connect(self.buttonClick)
        
        # Creating VBox to layout the widgets we have
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

        #Using radio buttons to make the user choose which of the 3 models to use
        self.LenetRadio = QRadioButton('LeNet5')
        self.AlexNetRadio = QRadioButton('AlexNet')
        self.BriaNetRadio = QRadioButton('BriaNet')
        self.LenetRadio.setChecked(True)

        vbox = QVBoxLayout()
        vbox.addWidget(self.LenetRadio)
        vbox.addWidget(self.AlexNetRadio)
        vbox.addWidget(self.BriaNetRadio)
        groupbox.setLayout(vbox)
        return groupbox

    def createParameters(self):

        groupbox = QGroupBox('Change Hyperparameters')

        ## Creating the epoch spinbox
        self.epochBox = QSpinBox()
        self.epochBox.setRange(5,30)
        epochLabel = QLabel("Number of epochs: (5 - 30 Range)")
        epochHbox = QHBoxLayout()
        epochHbox.addWidget(epochLabel)
        epochHbox.addWidget(self.epochBox)

        epochHbox.addStretch(1)

        ## Creating the batch_size spinbox
        batchLabel = QLabel("Batch Size: (30 - 128 Range)")
        self.batchBox = QSpinBox()
        self.batchBox.setRange(30,128)
        batchHbox = QHBoxLayout()
        batchHbox.addWidget(batchLabel)
        batchHbox.addWidget(self.batchBox)
        batchHbox.addStretch(1)
        
        ## Creating the learning_rate Spinbox (specified as double so we can use floats)
        learnLabel = QLabel("Learning Rate: (0.001 - 0.1 Range)")
        self.learnBox = QDoubleSpinBox()
        self.learnBox.setRange(0.001,0.1)
        self.learnBox.setSingleStep(0.001)
        self.learnBox.setDecimals(3)
        learnHbox = QHBoxLayout()
        learnHbox.addWidget(learnLabel)
        learnHbox.addWidget(self.learnBox)
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
        
        groupbox = QGroupBox('Train And View Dataset')
        

        # Start training button, along with a LineEdit for the user to type in a custom .pth file name
        startTrainButton = QPushButton('Train Model using selected dataset')
        self.modelLineEdit = QLineEdit()
        startTrainButton.clicked.connect(self.trainButtonClicked)
        
        self.modelLineEdit.setPlaceholderText('model1')
        self.modelLineEdit.setText('Give a name of saved model')
        viewImageButton = QPushButton('View images from dataset')

        ## Creating the filter combobox, user can select between either All letters, or just one image shown
        self.cb = QComboBox(self)
        self.cb.addItem('All')
        for letter in string.ascii_uppercase:
            if (letter != 'J') and (letter != 'Z'):
                self.cb.addItem(letter)

        hbox1 = QHBoxLayout()
        hbox1.addWidget(viewImageButton)
        hbox1.addWidget(self.cb)
        hbox2 = QHBoxLayout()
        hbox2.addWidget(startTrainButton)
        hbox2.addWidget(self.modelLineEdit)
        vbox = QVBoxLayout()
        vbox.addLayout(hbox2)
        self.datasetBrowser = QTextBrowser()
        vbox.addWidget(self.datasetBrowser)
        vbox.addStretch(1)
        vbox.addLayout(hbox1)
        
        groupbox.setLayout(vbox)
        viewImageButton.clicked.connect(self.openDatasetViewer)
        return groupbox
    
    def buttonClick(self):

        ## When the button is clicked, we open the file dialog to make the user choose a csv file as their dataset
        self.nameFile= QFileDialog.getOpenFileName(self,"Open training dataset",r"<Default dir>", "CSV (*.csv);;All Files (*)")[0]
        self.label1.setText("Directory: \n" + self.nameFile)
        self.label1.adjustSize()       # adjust the label size automatically
        self.trainGroupBox.setEnabled(True)
        self.datasetBrowser.clear()
        xy = np.genfromtxt(self.nameFile, delimiter = ",", dtype = np.uint8)[1:,0]
        zeros = np.zeros(25)
        for label in xy:
            zeros[label] +=1
        for i,num in enumerate(zeros):
            letter = string.ascii_uppercase[i]
            if letter != 'J':
                self.datasetBrowser.append(letter + ': '+str(int(num)) + ' counts')
                QApplication.processEvents()
            
    
    def sliderChange(self):
        self.sliderText.setText('Train/Validation ratio: ' + str(self.slider.value()) + '/' + str(100-self.slider.value()))
        self.sliderText.adjustSize()
    

    def openDatasetViewer(self):
        ## Creating instance of the imageGallery class, as well as passing through the filter (self.cb.currentText())
        self.w = imageGallery.imageViewer(str(self.nameFile),(self.cb.currentText()))
        
        self.w.show()
    def trainButtonClicked(self):

        ## Getting the model name and removing whitespaces
        self.modelName = self.modelLineEdit.text()
        self.modelName = self.modelName.strip()

        ## If the modelname was left empty we just rename to model1.pth
        if self.modelName == '':
            self.modelName = 'model1'

        # The training process (Could potentially have been put into a function itself)
        #Setting up has an parameter for AlexNet since it has different image set up requirements
        self.train = Training.Test_Train(batch_size=self.batchBox.value(),learning_rate=self.learnBox.value(),num_epochs=self.epochBox.value(),num_classes=26)
        self.device, self.train_dataset, self.valid_dataset = self.train.setting_up(file_location_train=self.nameFile, num_split=self.slider.value(), AlexNet = self.AlexNetRadio.isChecked())
        self.train_load, self.valid_load = self.train.loading_up(self.train_dataset, self.valid_dataset)
        
        # Checking which model was checked
        if self.LenetRadio.isChecked():
            self.model = self.train.runModel(self.train_load, self.valid_load, self.device, modelType='LeNet5')
        elif self.AlexNetRadio.isChecked():
            self.model = self.train.runModel(self.train_load, self.valid_load, self.device, modelType='AlexNet')
        else:
            self.model = self.train.runModel(self.train_load, self.valid_load, self.device, modelType='CNN')
        # Saving the new model
        torch.save(self.model,(self.modelName + '.pth'))
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = trainWindow()
    ex.show()
    sys.exit(app.exec_())
