import torch
import numpy as np
from CNN_model import CNN, nn
from dataset import userData
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QProgressBar
from tqdm import tqdm #for progress bar on console
from GUI_loading import MyApp

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

model = CNN(num_classes)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, weight_decay = 0.005, momentum = 0.9)

total_step = len(train_loader)

#Saving and loading of model. The print function within for loop (lines 81 and 82) to see the tensors within the trained model
#FILE = "modelV1.pth"
#torch.save(model.state_dict(), FILE)    #Can uncomment this if u like. Refer to this link https://www.youtube.com/watch?v=9L9jEOwRrCg for directions on how to save on CPU or GPU or both
#loaded_model = CNN(num_classes)
#loaded_model.load_state_dict(torch.load(FILE))
#loaded_model.eval()

#for param in model.parameters():
#    print(param)

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

    print('Accuracy of the network on the {} train images: {} %'.format(27455, 100*correct/total))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MyApp()
    #initParam(50, 26, 0.001, 20)
    sys.exit(app.exec_())