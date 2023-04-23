import cv2
import torch
import torch
import torchvision.transforms as transforms
import numpy as np
import sys
import string

from PyQt5.QtWidgets import*
from PyQt5.QtGui import*
from PyQt5.QtCore import*

#Model is a loaded saved model file. 
def testModel(model,image, useAlexNet):
    model.eval()
    input_image = cv2.imread(image)
   
    input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('image.png',input_image_gray)
    
    #------Need to add a specific transform parameters for AlexNet----------
    # If its AlexNet the resize will be of a different size (224x224) in comparison to LeNet5 and BriaNet
    if useAlexNet == True:
        processingImg = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(1), 
                                        transforms.Resize((224,224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = (0.4914,), std = (0.2023,))])
    else:
        processingImg = transforms.Compose([transforms.ToPILImage(), transforms.Grayscale(1), 
                                        transforms.Resize((32,32)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean = (0.1306,), std = (0.3082,))])
    
    input_tensor = processingImg(input_image_gray)
    
    input_batch = input_tensor.unsqueeze(0)
    

    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
    
    with torch.no_grad():
        output = model(input_batch)


  
    probabilities = torch.nn.functional.softmax(output, dim = 1)

    guess = np.argmax(probabilities)
  
    return probabilities, guess
    
    
class TestResults(QWidget):

    def __init__(self,model,images, useAlexNet):
        super().__init__()
        self.images = images
        self.model = model
        self.imageIndex = 0
        self.useAlexNet = useAlexNet #Boolean 
        self.initUI()
        self.startTest()

    def startTest(self):
        self.clear_text()
        image = self.images[self.imageIndex]
        high_res = QSize(320,320)
        self.currentImage.setPixmap(QPixmap(image).scaled(high_res))

        #Passes a boolean to differentiate whether it is AlexNet operation or not
        probs, guess = testModel(model=self.model,image=image, useAlexNet= self.useAlexNet)
        probs = probs.squeeze().tolist()
        # For each letter, we write down the % prediction of it
        for i,val in enumerate(probs):

            self.tb.append(string.ascii_uppercase[i] + ': ' + str(round((val * 100),2)) + '% accuracy')
        # Output our best guess (highest prediction value)
        self.tb.append('The best guess we have for this image is: '+ string.ascii_uppercase[(guess.tolist())])
        
        
        # Checking whether or not we are on the first or last image in the bunch (to enable/disable buttons)    
        if (self.imageIndex == 0):
            self.backButton.setEnabled(False)
        else:
            self.backButton.setEnabled(True)
        self.imageIndex+=1
        if self.imageIndex == len(self.images):
            self.newImageButton.setEnabled(False)
        else:
            self.newImageButton.setEnabled(True)
                
    def initUI(self):  
        # Creating the two main groupboxes for the image and the predictions section
        imgGroupBox = QGroupBox('Image being tested')
        probabilityGroupBox = QGroupBox('Probabilities')
        self.currentImage = QLabel()
        self.tb = QTextBrowser()
        self.tb.setAcceptRichText(True)
        
        # Creating the vbox for the image section
        imageVbox = QVBoxLayout()
        imageVbox.addWidget(self.currentImage)
        imgGroupBox.setLayout(imageVbox)
        probVbox = QVBoxLayout()
        probVbox.addWidget(self.tb)
        
        # Creating the section for the predictions and the image maneuvering section 
        self.newImageButton = QPushButton('Next Image')
        self.backButton = QPushButton('Previous Image')
        buttonHbox = QHBoxLayout()
        buttonHbox.addWidget(self.newImageButton)
        buttonHbox.addWidget(self.backButton)
        probVbox.addLayout(buttonHbox)
        probabilityGroupBox.setLayout(probVbox)
        Mainhbox = QHBoxLayout()
        self.backButton.setEnabled(False)
        Mainhbox.addWidget(imgGroupBox, 1)
        Mainhbox.addWidget(probabilityGroupBox,2)

        self.setLayout(Mainhbox)
        self.newImageButton.clicked.connect(self.startTest)
        self.backButton.clicked.connect(self.back)
        self.setWindowTitle('Image Predictions')
        self.setGeometry(300, 300, 800, 550)
        #self.show()

    def clear_text(self):
        # Clearing the text
        self.tb.clear()
        
    def back(self): 
        self.imageIndex = self.imageIndex-2
        self.startTest()
       

if __name__ == '__main__':
    app = QApplication(sys.argv)
    model = torch.load('properModelV1.pth')
    ex = TestResults(model=model,images=['/Users/shaaranelango/Downloads/project-1-python-team_16/T.jpg','/Users/shaaranelango/Downloads/project-1-python-team_16/W.jpg'])
    ex.show()
    sys.exit(app.exec_())
