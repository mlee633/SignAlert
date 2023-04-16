import torch
from dataset import userData
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from CNN_model import CNN

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
    
    def runModel(self, train_loader, test_loader, device):
        model = CNN(self.num_classes)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.SGD(model.parameters(), lr = self.learning_rate, weight_decay = 0.005, momentum = 0.9)

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
                MyApp.step = epoch #how to set the progress bar to move for GUI without having to import from the GUI file itself since GUI also requires to import this entire file to run the train/test code

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
