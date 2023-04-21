import cv2
import torch
import torch
import torchvision.transforms as transforms
import numpy as np
import sys
from PyQt5.QtWidgets import*
from PyQt5.QtGui import*
from PyQt5.QtCore import*

def testModel(model,image):
    model.eval()
    input_image = cv2.imread(image)
   
    input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    #input_image_gray.resize(32,32)
    cv2.imwrite('image.png',input_image_gray)
    #print(input_image_gray)
    processingImg = transforms.Compose([
    transforms.ToPILImage(), transforms.Grayscale(1),           
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize(mean = (0.1306,), std = (0.3082,))])
    input_tensor = processingImg(input_image_gray)
    
    input_batch = input_tensor.unsqueeze(0)
    t = transforms.ToPILImage()
    im = t(input_image_gray)
    im.save('return.png')
    
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    
    with torch.no_grad():
        output = model(input_batch)


    print(output)
    probabilities = torch.nn.functional.softmax(output, dim = 1)
    print(probabilities)
    probabilities = np.argmax(probabilities)
    print(probabilities)
    
class TestResults(QWidget):

    def __init__(self,model,images):
        super().__init__()
        self.initUI()
        self.images = images
        self.model = model


    def initUI(self):  
        imgGroupBox = QGroupBox('Image being tested')
        probabilityGroupBox = QGroupBox('Probabilities')
        self.currentImage = QLabel()
        self.tb = QTextBrowser()
        self.tb.setAcceptRichText(True)
        
        
        imageVbox = QVBoxLayout()
        imageVbox.addWidget(self.currentImage)
        imgGroupBox.setLayout(imageVbox)
        probVbox = QVBoxLayout()
        probVbox.addWidget(self.tb)
        probabilityGroupBox.setLayout(probVbox)
        Mainhbox = QHBoxLayout()
        
        Mainhbox.addWidget(imgGroupBox, 1)
        Mainhbox.addWidget(probabilityGroupBox,2)

        self.setLayout(Mainhbox)

        self.setWindowTitle('QTextBrowser')
        self.setGeometry(300, 300, 600, 300)
        self.show()

    def clear_text(self):
        self.tb.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = TestResults('x','image.png')
    sys.exit(app.exec_())
