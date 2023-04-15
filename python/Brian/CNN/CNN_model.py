
import torch.nn as nn
from tqdm import tqdm #for progress bar on console
import torch
import numpy as np
from dataset import userData
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from GUI_loading import MyApp

"""batch_size = 50
num_classes = 26
learning_rate = 0.001
num_epochs = 20

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

train_dataset = userData('C:\\Users\\OEM\\Downloads\\archive\sign_mnist_train\\sign_mnist_train.csv',
                         transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize((32,32)),
                                                        transforms.Normalize(mean = (0.1306,), std = (0.3082,))]))

test_dataset = userData('C:\\Users\\OEM\\Downloads\\archive\sign_mnist_test\\sign_mnist_test.csv',
                         transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize((32,32)),
                                                        transforms.Normalize(mean = (0.1306,), std = (0.3082,))]))

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False) """ 
#will add a parameter for the csv file directory
def initParam(batch_size, num_classes, learning_rate, num_epochs):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
    train_dataset = userData('C:\\Users\\OEM\\Downloads\\archive\sign_mnist_train\\sign_mnist_train.csv',
                         transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize((32,32)),
                                                        transforms.Normalize(mean = (0.1306,), std = (0.3082,))]))

    test_dataset = userData('C:\\Users\\OEM\\Downloads\\archive\sign_mnist_test\\sign_mnist_test.csv',
                         transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize((32,32)),
                                                        transforms.Normalize(mean = (0.1306,), std = (0.3082,))]))
    train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False) 

    model = CNN(num_classes)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 0.005, momentum = 0.9)

    #total_step = len(train_loader)

    for epoch in range(num_epochs):
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
            MyApp.step = epoch

        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

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
    return 

class CNN(nn.Module):

    #input channels = the colour of image. 3 for colour, 1 for greyscale
    #out channels = number of features. Seems like this out can be any number u choose. 
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        #oiriginal image is 32 x 32
        self.c1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3) #image becomes 30 x 30
        self.c2 = nn.Conv2d(in_channels= 32, out_channels = 32, kernel_size = 3) #image becomes 28 x 28 but with 32 channels???
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2) #image 28 x 28
        #kernal_size = a n x m amount of pixels used to create a pixel for the filtered image. Basically the amount use to filer an image
        #Stride = amount of pixels moved to the right from the starting position of the kernal

        self.c3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3) # image becomes 26 x 26
        self.c4 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size = 3) #image becomes 24 x 24
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2) # image becomes 24 x 24
        
        self.fc1 = nn.Linear(1600, 128) #first value seems to be final number of channels * final image size
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.c1(x)
        out = self.c2(out)
        out = self.max_pool1(out)

        out = self.c3(out)
        out = self.c4(out)
        out = self.max_pool2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out
    
"""model = CNN(num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 0.005, momentum = 0.9)

#total_step = len(train_loader)

#Saving and loading of model. The print function within for loop (lines 81 and 82) to see the tensors within the trained model
#FILE = "modelV1.pth"
#torch.save(model.state_dict(), FILE)    #Can uncomment this if u like. Refer to this link https://www.youtube.com/watch?v=9L9jEOwRrCg for directions on how to save on CPU or GPU or both
#loaded_model = CNN(num_classes)
#loaded_model.load_state_dict(torch.load(FILE))
#loaded_model.eval()

for param in model.parameters():
    print(param)
#====Apparently lazy method of saving models===
#FILE = "modelV1.pth"
#torch.save(model, FILE)
#model = torch.load(FILE)
#model.eval()

#Uncomment all below this if wanting to train the model again from scratch. 
#tqdm for the progress bar stuff on console. 
for epoch in range(num_epochs):
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
        MyApp.step = epoch

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

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

    print('Accuracy of the network on the {} train images: {} %'.format(27455, 100*correct/total)) """