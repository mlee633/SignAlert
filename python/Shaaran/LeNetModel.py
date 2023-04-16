import torch 
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np

from userDataset import userData
from PIL import Image
# Define relevant variables for the ML task #
batch_size = 50
num_classes = 26
learning_rate = 0.0001
num_epochs = 15

# Device will determine whether to run the training on GPU or CPU. #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Loading the dataset + preprocessing ###
## Data is transformed to 32x32 because the LeNet5 model uses a 32x32 input ##

# Training Dataset
train_dataset = userData('/Users/shaaranelango/Downloads/project-1-python-team_16/dataset/sign_mnist_train.csv',transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),transforms.Normalize(mean = (0.1306,), std = (0.3082,))]))


# Testing Dataset
test_dataset = userData('/Users/shaaranelango/Downloads/project-1-python-team_16/dataset/sign_mnist_test.csv',transform=transforms.Compose([transforms.ToTensor(),transforms.Resize((32,32)),transforms.Normalize(mean = (0.1305,), std = (0.3084,))]))
# Loading the trainer
train_loader = DataLoader(dataset = train_dataset,batch_size = batch_size,shuffle = True)

# Loading the tester
test_loader = DataLoader(dataset = test_dataset,batch_size = batch_size,shuffle = False)

#Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out

model = LeNet5(num_classes).to(device)

#Setting the loss function
cost = nn.CrossEntropyLoss()

#Setting the optimizer with the model parameters and learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#this is defined to print how many steps are remaining when training

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # images.numpy()
        # images = Image.fromarray(images)
        # images = images.convert("L")
        labels = labels.T
        labels = np.ravel(labels)
        labels = torch.from_numpy(labels)
        images = images.to(device)
        labels = labels.to(device)
        #print(labels)
        optimizer.zero_grad()
        #Forward pass
        outputs = model(images)
        loss = cost(outputs, labels)
        
        
        # Backward and optimize
        
        loss.backward()
        optimizer.step()
        		
        if (i+1) % 50 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
        		           .format(epoch+1, num_epochs, i+1, total_step, loss.item()))


# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
  
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

    print('Accuracy of the network on the {} test images: {}%'.format(test_dataset.n_samples,100 * correct / total))
	 