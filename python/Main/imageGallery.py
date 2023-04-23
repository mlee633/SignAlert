import sys
import string
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import*
import numpy as np
from PIL import Image, ImageQt

## Creating a subclass of QAbstractListModel so that we can properly display the images in the gallery through a list model
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
        ## Dictionary with all the letters and it's associate index label value
        self.labelLetters = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24}
        self.dataset = Dataset
        self.filter = filter
        ## Creating the QListView and giving it the QAbstractListModel model
        self.listview = QListView(self)
        self.model = imageModel(self)
        self.listview.setModel(self.model)
        self.listview.setFlow(0)
        self.listview.setWrapping(True)
        self.initUI()

    def initUI(self):
        data = np.genfromtxt(self.dataset, delimiter=',')[1:, :] #Remove first row (pixel numbers)
        imageLabels = data[:,0]
        

        for label, image in zip(imageLabels, data[:,1:]):
            # If the filter is for all images, or if the label is the same as the filter
            if (self.filter== 'All') or (self.labelLetters.get(self.filter) == label):
                # Reshape the csv file to 28x28 array and turn it into a PIL image then to a QtImage
                image = image.reshape(28, 28)
                im = Image.fromarray(image)
                im = im.convert("L")
                im = ImageQt.ImageQt(im)
                pixmap = QPixmap.fromImage(im)
                # Scaling up the image so that it looks better on the gallery
                pixmap = pixmap.scaled(140, 140)
                # Appending the image to the listView
                self.model.images.append(pixmap)
                self.model.layoutChanged.emit()
        self.setCentralWidget(self.listview)
        self.setWindowTitle('Dataset Image Viewer')
        self.centre()
    
    ## Centering the application
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