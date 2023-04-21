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
import wandb
#from python.Shaaran.PyQtGUI.progressBar import PBar

## Found off internet as a Transform to add Guassian Noise to make the dataset less overfitted (https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745)
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
    
wandb.init(
    # set the wandb project where this run will be logged
    project="my-awesome-project",
    group = "testing 1", 
    name = "CNN graph?",
    
    # track hyperparameters and run metadata
    config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

def top_k_accuracy(scores, labels, topk=(1, )):
    """Calculate top k accuracy score.
    Args:
        scores (list[np.ndarray]): Prediction scores for each class.
        labels (list[int]): Ground truth labels.
        topk (tuple[int]): K value for top_k_accuracy. Default: (1, ).
    Returns:
        list[float]: Top k accuracy score for each k.
    """
    res = np.zeros(len(topk))
    labels = np.array(labels)[:, np.newaxis]
    for i, k in enumerate(topk):
        max_k_preds = np.argsort(scores, axis=1)[:, -k:][:, ::-1]
        match_array = np.logical_or.reduce(max_k_preds == labels, axis=1)
        topk_acc_score = match_array.sum() / match_array.shape[0]
        res[i] = topk_acc_score

    return res


class Test_Train:
    def __init__(self, batch_size, num_classes, learning_rate, num_epochs):
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.progressBar = MyApp() #PBar()

    #num_split only in range from 0 to 1
    def setting_up(self, filename, num_split):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        split = num_split/100
        xy = np.genfromtxt(filename, delimiter = ",", dtype = np.uint8)[1:,:]
        print(xy.shape[0])
        train_dataset = xy[(int(split*xy.shape[0])),:] #numpy array
        valid_dataset = xy[(int((1-split)*xy.shape[0])),:]
        train_dataset = userData(train_dataset, transform=transforms.Compose([transforms.ToTensor(), transforms.RandomRotation(30),
                                                        transforms.Resize((32,32)),
                                                        transforms.Normalize(mean = (0.1306,), std = (0.3082,))]))
        valid_dataset = userData(valid_dataset, transform=transforms.Compose([transforms.ToTensor(),
                                                        transforms.Resize((32,32)),
                                                        transforms.Normalize(mean = (0.1306,), std = (0.3082,))]))
        return device, train_dataset, valid_dataset 
    
    def loading_up(self, train_dataset, test_dataset):
        train_loader = DataLoader(dataset = train_dataset, batch_size = self.batch_size, shuffle = True)
        test_loader = DataLoader(dataset = test_dataset, batch_size = 1, shuffle = True) 
    
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
        optimizer  = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=0.01)

        #used for debugging
        # for param in model.parameters():
            #print(param.size())

        #Training along with accuracy of model
        counter = 0
        loop = 0
        for epoch in range(self.num_epochs):
            running_loss = 0
            for i, (images, labels) in enumerate(train_loader): #tqdm(enumerate(train_loader), total = len(train_loader), leave = False):
                if self.progressBar.action == True:
                    labels = labels.T
                    labels = np.ravel(labels)
                    labels = torch.from_numpy(labels)
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images) 
                    loss = criterion(outputs,labels)
                    
                    running_loss += loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    counter += 1
                    self.progressBar.updateProgress(int(100 *counter / (self.num_epochs * len(train_loader))))
                    avg_loss = running_loss / (i+1)
                else:
                    return #Stops training when button is clicked on gui
                
            #Calculates the accuracy of the model for every epoch loop
            running_vloss = 0
            running_acc = np.zeros(3)
            with torch.no_grad():
                correct = 0
                total = 0    
                for i, (images, labels) in enumerate(test_loader):
                    #print(labels.shape)
                    outputs = model(images)
                    #print(outputs.shape)
                    running_acc += top_k_accuracy(outputs.detach().cpu().numpy(),
                                           labels.detach().cpu().numpy(), topk=(1, 3, 5))

                    labels = labels.T
                    labels = np.ravel(labels)
                    labels = torch.from_numpy(labels)
                    images = images.to(device)
                    labels = labels.to(device)
                    #outputs = model(images)
                    vloss = criterion(outputs,labels)
                    running_vloss+= vloss
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    avg_vloss = running_vloss / (i + 1)
                    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))


            #Displays this onto the textbox that is located above the progress bar. Also seems like number of train images are len(train_loader)
            loop += 1
            print("Loop Num: {}=========================================================".format(loop))
            self.progressBar.tb.append('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, self.num_epochs, loss.item()))
            self.progressBar.tb.append('Accuracy of the network on the {} train images: {:.2f} % \n'.format(27455, 100*correct/total))
            running_acc /= len(test_loader)
            wandb.log({'train/loss': avg_loss,
                'train/learning_rate': self.learning_rate,
                'val/top1_accuracy': running_acc[0].item(),
                'val/top3_accuracy': running_acc[1].item(),
                'val/top5_accuracy': running_acc[2].item(),
                'Shaaran/avg_vloss' : avg_vloss,
                'Shaaran/correct_prediction': correct})
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
    stupid = Test_Train(55, 26, 0.001, 20)
    device, train_dataset, test_dataset = stupid.setting_up('C:\\Users\\brian\Documents\\project-1-python-team_16\\dataset\\sign_mnist_train.csv', 50)
    train_load, test_load = stupid.loading_up(train_dataset, test_dataset)
    model = stupid.runModel(train_load, test_load, device, "CNN")
    filename = "CNNV1"
    torch.save(model,(filename + '.pth'))
    """loaded_model = torch.load('CNNV1.pth')
    loaded_model.eval()
   
    input_image = cv2.imread("C:\\Users\\brian\\Downloads\\R.jpg")
    #input_image = c
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
    # print(len(input_batch))
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        loaded_model.to('cuda')
    
    with torch.no_grad():
        output = loaded_model(input_batch)


    print(output)
    probabilities = torch.nn.functional.softmax(output, dim = 1)
    print(probabilities)
    probabilities = np.argmax(probabilities)
    print(probabilities)


 #All stuff underneath is just the required file directory for testing on different devices. 
    #'C:\\Users\\brian\Documents\\project-1-python-team_16\\dataset\\sign_mnist_train.csv'
    #'C:\\Users\\brian\Documents\\project-1-python-team_16\\dataset\\sign_mnist_test.csv'
    #'C:\\Users\\OEM\\Downloads\\archive\sign_mnist_train\\sign_mnist_train.csv'
    #'C:\\Users\\OEM\\Downloads\\archive\sign_mnist_test\\sign_mnist_test.csv')
    #C:\\Users\\OEM\\Documents\\project-1-python-team_16\\python\\Brian\\CNN\\WIN_20230420_01_04_07_Pro.jpg
    #'C:\\Users\\brian\\Documents\\project-1-python-team_16\\python\\Brian\\CNN\\WIN_20230420_01_04_07_Pro.jpg'"""