import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from dataset import userData
from torch.utils.data import DataLoader
import numpy as np

batch_size = 50
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
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)

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
    
model = CNN(num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 0.005, momentum = 0.9)

#EPOCH = 20
#PATH = "model.pt"
#LOSS = 

total_step = len(train_loader)

for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
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

    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
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