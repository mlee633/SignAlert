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

        #for param in model.parameters():
            #print(param.size())
        counter = 0
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(train_loader): #tqdm(enumerate(train_loader), total = len(train_loader), leave = False):
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
                    return
                    #break
            #if self.progressBar.action == False:
            #    break
            #else:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_epochs, loss.item()))
        

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

            print('Accuracy of the network on the {} train images: {} %'.format(27455, 100*correct/total)) 


if __name__ == '__main__':
    app = QApplication(sys.argv)
    stupid = Test_Train(50, 26, 0.001, 20)
    device, train_dataset, test_dataset = stupid.setting_up('C:\\Users\\OEM\\Downloads\\archive\\sign_mnist_train\\sign_mnist_train.csv', 'C:\\Users\\OEM\\Downloads\\archive\\sign_mnist_test\\sign_mnist_test.csv', )
    train_load, test_load = stupid.loading_up(train_dataset, test_dataset)
    stupid.runModel(train_load, test_load, device, "CNN")