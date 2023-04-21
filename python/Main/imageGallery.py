import sys
import string
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import*
import numpy as np
from PIL import Image, ImageQt

class imageModel(QAbstractListModel):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.images = []
    def data(self, index: QModelIndex, role):
        row = index.row()
        if role == Qt.DecorationRole:
            return self.images[row]
    def rowCount(self, parent=None):
        return len(self.images)

class imageViewer(QMainWindow):
    def __init__(self,Dataset,filter):
        super().__init__()
        self.labelLetters = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24}
        self.dataset = Dataset
        self.filter = filter
        self.listview = QListView(self)
        self.model = imageModel(self)
        self.listview.setModel(self.model)
        self.listview.setFlow(0)
        self.listview.setWrapping(True)
        self.initUI()
    def initUI(self):
        data = np.genfromtxt(self.dataset, delimiter=',')[1:, :] #Remove first row (pixel numbers)
        imageLabels = data[:,0]
        # Label isn't needed right now
        for label, image in zip(imageLabels, data[:,1:]):
            if (self.filter== 'All') or (self.labelLetters.get(self.filter) == label):
                image = image.reshape(28, 28)
                im = Image.fromarray(image)
                im = im.convert("L")
                im = ImageQt.ImageQt(im)
                pixmap = QPixmap.fromImage(im)
                pixmap = pixmap.scaled(140, 140)
                self.model.images.append(pixmap)
                self.model.layoutChanged.emit()
        self.setCentralWidget(self.listview)
        self.setWindowTitle('Dataset Image Viewer')
        self.centre()
        #self.show()
    def centre(self):
        self.setGeometry(0,0,770,800)
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = imageViewer('/Users/shaaranelango/Downloads/project-1-python-team_16/dataset/sign_mnist_train.csv','A')
    sys.exit(app.exec_())