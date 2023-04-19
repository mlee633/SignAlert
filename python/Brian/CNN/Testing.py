import torch
from dataset import userData
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from CNN_model import CNN
from GUI_loading import MyApp
from PyQt5.QtWidgets import QApplication
import sys
from tqdm import tqdm
from torch.utils.data import DataLoader
import time

class Test_Train:
    def __init__(self, batch_size, num_classes, learning_rate, num_epochs):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

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
    
    def runModel(self, device):
        #Remember to add in if statements to allow users to change models here. 
        model = CNN(self.num_classes).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr = self.learning_rate, weight_decay = 0.005, momentum = 0.9)

        app = QApplication(sys.argv)
        gui = MyApp()

        #To check the size, size of input and output channels. 
        #for param in model.parameters():
           # print(param.size())

        return model, criterion, optimizer, app, gui
    
    def training_model(self, model, criterion, optimizer, train_loader, device, app, gui):
        counter = 0
        for epoch in range(self.num_epochs):
            for i, (images, labels) in enumerate(train_loader): #tqdm(enumerate(train_loader), total = len(train_loader), leave = False):
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
                #gui.pbar.setValue(gui.pbar.value()+ 1/(self.num_epochs * self.batch_size))
                #gui.show()
                counter += 1
                gui.pbar.setValue(int(100* counter/ (self.num_epochs * len(train_loader)))) #using the tqdm values to make the progress bar. I is the number of cycles done?
                

            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_epochs, loss.item()))
            
        sys.exit(app.exec_())
        

    def testing_model(self, test_loader, device, model):
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
    stupid = Test_Train(50, 26, 0.001, 20)
    device, train_set, test_set = stupid.setting_up('C:\\Users\\brian\Documents\\project-1-python-team_16\\dataset\\sign_mnist_train.csv', 'C:\\Users\\brian\Documents\\project-1-python-team_16\\dataset\\sign_mnist_test.csv')
    train_load, test_load = stupid.loading_up(train_set, test_set)
    model, criterion, optimizer, app, gui = stupid.runModel(device)
    stupid.training_model(model, criterion, optimizer, train_load, device, app, gui)

    #All stuff underneath is just the required file directory for testing on different devices. 
    #'C:\\Users\\brian\Documents\\project-1-python-team_16\\dataset\\sign_mnist_train.csv'
    #'C:\\Users\\brian\Documents\\project-1-python-team_16\\dataset\\sign_mnist_test.csv'
    #'C:\\Users\\OEM\\Downloads\\archive\sign_mnist_train\\sign_mnist_train.csv'
    #'C:\\Users\\OEM\\Downloads\\archive\sign_mnist_test\\sign_mnist_test.csv')
    

    #Add the signal emitting for the thread class within the for loop of epoch??
    #and see if this would work as the same as when testing on the GUI file. Rather than calling
    #via threadclass on Gui gile, you create instance of threadclass within the epoch calculating method. 