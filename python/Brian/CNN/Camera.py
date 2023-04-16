import sys
from PyQt5.QtWidgets import QApplication, QWidget, QProgressBar, QPushButton, QVBoxLayout
from PyQt5.QtCore import Qt, QThread, pyqtSignal

class WorkerThread(QThread):
    progress_update = pyqtSignal(int)

    def run(self):
        for i in range(101):
            self.progress_update.emit(i)
            self.msleep(50)

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.title = 'Progress Bar Example'
        self.left = 100
        self.top = 100
        self.width = 400
        self.height = 200
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(30, 40, 350, 25)
        self.progress_bar.setValue(0)

        self.train_button = QPushButton('Train Data', self)
        self.train_button.clicked.connect(self.start_training)
        self.train_button.setGeometry(150, 100, 100, 25)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.progress_bar)
        self.layout.addWidget(self.train_button)
        self.setLayout(self.layout)

        self.show()

    def start_training(self):
        self.thread = WorkerThread()
        self.thread.progress_update.connect(self.update_progress_bar)
        self.thread.start()

    def update_progress_bar(self, progress):
        self.progress_bar.setValue(progress)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())