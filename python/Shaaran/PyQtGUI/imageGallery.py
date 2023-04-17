import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import numpy as np
from PIL import Image, ImageQt

class MainWindow(QWidget):
    def __init__(self,Dataset,filter):
        self.labelLetters = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'K':10,'L':11,'M':12,'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24}
        super().__init__()
        self.setWindowTitle('Images from dataset')
        self.setGeometry(400, 400, 400, 800)

        formLayout = QFormLayout()
        groupBox = QGroupBox()
        data = np.genfromtxt(Dataset, delimiter=',')[1:, :] #Remove first row (pixel numbers)
        imageLabels = data[:,0]
        if filter == None:
            i=0
            for label in imageLabels:
                label1 = QLabel(self.labelLetters.get(label))
                label2 = QLabel()
                image = data[i,1:]
                image = image.reshape(28,28)
                im = Image.fromarray(image)
                im = im.convert("L")
                im = ImageQt.ImageQt(im)
                label2.setPixmap(QPixmap.fromImage(im))
                formLayout.addRow(label1, label2)
                i+=1
           
        # for n in range(100):
        #     label1 = QLabel('Slime_%2d' % n)
        #     label2 = QLabel()
        #     label2.setPixmap(QPixmap('s1.png'))
        #     formLayout.addRow(label1, label2)

        groupBox.setLayout(formLayout)

        scroll = QScrollArea()
        scroll.setWidget(groupBox)
        scroll.setWidgetResizable(True)

        layout = QVBoxLayout(self)
        layout.addWidget(scroll)


if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow('/Users/shaaranelango/Downloads/project-1-python-team_16/dataset/sign_mnist_train.csv',None)
    window.show()
    sys.exit(app.exec())
# import sys
# from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QComboBox, QHBoxLayout, QVBoxLayout, QScrollArea, QWidget
# from PyQt5.QtGui import QPixmap
# from PyQt5.QtCore import Qt
# import pandas as pd
# from collections import Counter
 
# class DatasetViewer(QMainWindow):
#     def __init__(self, data_path):
#         super().__init__()
#         self.data_path = data_path
#         self.dataset = pd.read_csv(self.data_path)
#         self.class_names = self.dataset['ClassId'].unique()
#         self.initUI()
 
#     def initUI(self):
#         self.setWindowTitle("Dataset Viewer")
 
#         self.image_label = QLabel()
#         self.image_label.setAlignment(Qt.AlignCenter)
 
#         self.class_combo = QComboBox()
#         self.class_combo.addItems([str(class_id) for class_id in self.class_names])
#         self.class_combo.currentIndexChanged.connect(self.filter_class)
 
#         self.count_label = QLabel()
#         self.update_count_label()
 
#         self.scroll_area = QScrollArea()
#         self.scroll_area.setWidgetResizable(True)
#         self.scroll_widget = QWidget()
#         self.scroll_widget_layout = QVBoxLayout()
#         self.scroll_widget.setLayout(self.scroll_widget_layout)
#         self.scroll_area.setWidget(self.scroll_widget)
 
#         main_layout = QVBoxLayout()
#         main_layout.addWidget(self.image_label)
#         main_layout.addWidget(self.class_combo)
 
#         stats_layout = QHBoxLayout()
#         stats_layout.addWidget(QLabel("Class Counts:"))
#         stats_layout.addWidget(self.count_label)
#         stats_widget = QWidget()
#         stats_widget.setLayout(stats_layout)
 
#         main_layout.addWidget(stats_widget)
#         main_layout.addWidget(self.scroll_area)
 
#         central_widget = QWidget()
#         central_widget.setLayout(main_layout)
#         self.setCentralWidget(central_widget)
 
#         self.showMaximized()
 
#     def update_count_label(self):
#         class_counts = Counter(self.dataset['ClassId'])
#         count_str = ''
#         for class_id in self.class_names:
#             count_str += f"{class_id}: {class_counts[class_id]}, "
#         count_str = count_str[:-2]  # remove trailing comma and space
#         self.count_label.setText(count_str)
 
#     def filter_class(self):
#         class_id = int(self.class_combo.currentText())
#         filtered_df = self.dataset[self.dataset['ClassId'] == class_id]
#         self.update_count_label()
 
#         self.scroll_widget_layout = QVBoxLayout()
#         for i in range(len(filtered_df)):
#             img_path = filtered_df.iloc[i]['Path']
#             pixmap = QPixmap(img_path)
#             pixmap = pixmap.scaledToWidth(200)
#             label = QLabel()
#             label.setPixmap(pixmap)
#             label.setAlignment(Qt.AlignCenter)
#             self.scroll_widget_layout.addWidget(label)
 
#         self.scroll_widget.setLayout(self.scroll_widget_layout)
 
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     viewer = DatasetViewer('/Users/shaaranelango/Downloads/project-1-python-team_16/dataset/sign_mnist_train.csv')
#     viewer.show()
#     sys.exit(app.exec_())