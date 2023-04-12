import sys
from PyQt5.QtWidgets import*


class trainWindow(QWidget):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        grid = QGridLayout()
        grid.addWidget(self.createFirstExclusiveGroup(), 0, 0)
        grid.addWidget(self.createSecondExclusiveGroup(), 1, 0)
        grid.addWidget(self.createNonExclusiveGroup(), 0, 1)
        grid.addWidget(self.createPushButtonGroup(), 1, 1)

        self.setLayout(grid)

        self.setWindowTitle('SignAlert - Train')
        self.setGeometry(300, 300, 700, 450)
        self.show()

    def createFirstExclusiveGroup(self):
        groupbox = QGroupBox('Choose Training Dataset (.csv format)')

        fileOpenButton = QPushButton('Select Dataset')
        fileOpenButton.clicked.connect(self.buttonClick)

        vbox = QVBoxLayout()
        vbox.addWidget(fileOpenButton)
        groupbox.setLayout(vbox)

        return groupbox

    def createSecondExclusiveGroup(self):
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

    def createNonExclusiveGroup(self):
        groupbox = QGroupBox('Non-Exclusive Checkboxes')
        groupbox.setFlat(True)                              # set different style of groupbox

        checkbox1 = QCheckBox('Checkbox1')
        checkbox2 = QCheckBox('Checkbox2')
        checkbox2.setChecked(True)
        tristatebox = QCheckBox('Tri-state Button')
        tristatebox.setTristate(True)

        vbox = QVBoxLayout()
        vbox.addWidget(checkbox1)
        vbox.addWidget(checkbox2)
        vbox.addWidget(tristatebox)
        vbox.addStretch(1)
        groupbox.setLayout(vbox)

        return groupbox

    def createPushButtonGroup(self):
        groupbox = QGroupBox('Push Buttons')
        groupbox.setCheckable(True)
        groupbox.setChecked(True)

        # different push buttons
        pushbutton = QPushButton('Normal Button')
        togglebutton = QPushButton('Toggle Button')
        togglebutton.setCheckable(True)
        togglebutton.setChecked(True)
        flatbutton = QPushButton('Flat Button')
        flatbutton.setFlat(True)
        popupbutton = QPushButton('Popup Button')
        menu = QMenu(self)
        menu.addAction('First Item')
        menu.addAction('Second Item')
        menu.addAction('Third Item')
        menu.addAction('Fourth Item')
        popupbutton.setMenu(menu)

        vbox = QVBoxLayout()
        vbox.addWidget(pushbutton)
        vbox.addWidget(togglebutton)
        vbox.addWidget(flatbutton)
        vbox.addWidget(popupbutton)
        vbox.addStretch(1)
        groupbox.setLayout(vbox)

        return groupbox
    def buttonClick(self):
        nameFile= QFileDialog.getOpenFileName(self,"Open training dataset",r"<Default dir>", "CSV (*.csv);;All Files (*)")
        print(nameFile)
        training_data = test.funct(nameFile[0])
        print(training_data.ndim)
        print(training_data[:, 0])
        print(training_data[1: , 1].ndim)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = trainWindow()
    sys.exit(app.exec_())
