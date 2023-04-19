import torch
from torch.utils.data import DataLoader
from dataset import userData
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from CNN_model import CNN
from GUI_loading import MyApp
from PyQt5.QtWidgets import QApplication
import sys
#from Isaac.PyGUI.AlexNet import AlexNet
from LeNet5Model import LeNet5
import cv2
#from python.Shaaran.PyQtGUI.progressBar import PBar

class Test_Train:
    def __init__(self, batch_size, num_classes, learning_rate, num_epochs):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.progressBar = MyApp() #PBar()

    def setting_up(self, file_location_train, file_location_test):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_dataset = userData(file_location_train,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize((32,32)),
                                                        transforms.Normalize(mean = (0.1306,), std = (0.3082,))]))
        test_dataset = userData(file_location_test,
                         transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize((32,32)),
                                                        transforms.Normalize(mean = (0.1306,), std = (0.3082,))]))
        return device, train_dataset, test_dataset 
    
    def loading_up(self, train_dataset, test_dataset):
        train_loader = DataLoader(dataset = train_dataset, batch_size = self.batch_size, shuffle = True)
        test_loader = DataLoader(dataset = test_dataset, batch_size = self.batch_size, shuffle = False) 
    
        return train_loader, test_loader
    
    def runModel(self, train_loader, test_loader, device, modelType):
        #Remember to add in if statements to allow users to change models here. 
        if modelType == 'CNN':
            model = CNN(self.num_classes).to(device)
        elif modelType == 'LeNet5':
            model = LeNet5(self.num_classes).to(device)
        #else:
        #    model = AlexNet(self.num_classes).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = self.learning_rate, weight_decay = 0.005, momentum = 0.9)

        #used for debugging
        # for param in model.parameters():
            #print(param.size())

        #Training along with accuracy of model
        counter = 0
        for epoch in range(self.num_epochs):
            for _, (images, labels) in enumerate(train_loader): #tqdm(enumerate(train_loader), total = len(train_loader), leave = False):
                if self.progressBar.action == True:
                    labels = labels.T
                    labels = np.ravel(labels)
                    labels = torch.from_numpy(labels)
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images) 
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    counter += 1
                    self.progressBar.updateProgress(int(100 *counter / (self.num_epochs * len(train_loader))))
                else:
                    return #Stops training when button is clicked on gui
                
            #Calculates the accuracy of the model for every epoch loop
            with torch.no_grad():
                correct = 0
                total = 0    
                for images, labels in test_loader:
                    labels = labels.T
                    labels = np.ravel(labels)
                    labels = torch.from_numpy(labels)
                    images = images.to(device)
                    labels = labels.to(device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            #Displays this onto the textbox that is located above the progress bar. Also seems like number of train images are len(train_loader)
            self.progressBar.tb.append('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_epochs, loss.item()))
            self.progressBar.tb.append('Accuracy of the network on the {} train images: {:.2f} % \n'.format(27455, 100*correct/total))
        return model

    def saving_model(self, trained_model, file_name):
        #if the method does return something, will save the model
        if trained_model is not None:
            torch.save(trained_model.state_dict(), file_name )
    
    def load_model(self, filename, modelType, device):
        if modelType == 'CNN':
            loaded_model = CNN(self.num_classes).to(device)
        elif modelType == 'LeNet5':
            loaded_model = LeNet5(self.num_classes).to(device)
        #else:
        #    loaded_model = AlexNet(self.num_classes).to(device)

        loaded_model.load_state_dict(torch.load(filename))
        return loaded_model
        #loaded_model.eval()

          
#For testing purposes when running on this file
if __name__ == '__main__':
    app = QApplication(sys.argv)
    stupid = Test_Train(50, 26, 0.001, 20)
    device, train_dataset, test_dataset = stupid.setting_up('C:\\Users\\brian\Documents\\project-1-python-team_16\\dataset\\sign_mnist_train.csv', 'C:\\Users\\brian\Documents\\project-1-python-team_16\\dataset\\sign_mnist_train.csv' )
    #train_load, test_load = stupid.loading_up(train_dataset, test_dataset)
    #model = stupid.runModel(train_load, test_load, device, "CNN")
    filename = "properModelV1"
    #stupid.saving_model(model, filename + ".pth")
    loaded_model = stupid.load_model("properModelV1.pth", "CNN", device)
    loaded_model.eval()
    input_image = cv2.imread('C:\\Users\\brian\\Documents\\project-1-python-team_16\\python\\Brian\\CNN\\WIN_20230420_01_04_07_Pro.jpg')
    input_image_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    #print(input_image_gray)
    processingImg = transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize((32,32)),
                                                        transforms.Normalize(mean = (0.1306,), std = (0.3082,))])
    input_tensor = processingImg(input_image_gray)
    input_batch = input_tensor.unsqueeze(0)
    #print(len(input_batch))
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        loaded_model.to('cuda')
    
    with torch.no_grad():
        output = loaded_model(input_batch)

    print(output)
    probabilities = torch.nn.functional.softmax(output[0], dim = 0)
    probabilities = np.argmax(probabilities)
    print(probabilities)


 #All stuff underneath is just the required file directory for testing on different devices. 
    #'C:\\Users\\brian\Documents\\project-1-python-team_16\\dataset\\sign_mnist_train.csv'
    #'C:\\Users\\brian\Documents\\project-1-python-team_16\\dataset\\sign_mnist_test.csv'
    #'C:\\Users\\OEM\\Downloads\\archive\sign_mnist_train\\sign_mnist_train.csv'
    #'C:\\Users\\OEM\\Downloads\\archive\sign_mnist_test\\sign_mnist_test.csv')
    #C:\\Users\\OEM\\Documents\\project-1-python-team_16\\python\\Brian\\CNN\\WIN_20230420_01_04_07_Pro.jpg
    #'C:\\Users\\brian\\Documents\\project-1-python-team_16\\python\\Brian\\CNN\\WIN_20230420_01_04_07_Pro.jpg'