
import torch.nn as nn
class CNN(nn.Module):

    #input channels = the colour of image. 3 for colour, 1 for greyscale
    #out channels = number of features. Seems like this out can be any number u choose. 
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        #oiriginal image is 32 x 32
        self.c1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 3) #image becomes 30 x 30
        self.c2 = nn.Conv2d(in_channels= 32, out_channels = 32, kernel_size = 3) #image becomes 28 x 28 but with 32 channels???
        self.max_pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2) #image 14 x 14
        #kernal_size = a n x m amount of pixels used to create a pixel for the filtered image. Basically the amount use to filer an image
        #Stride = amount of pixels moved to the right from the starting position of the kernal

        self.c3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 3) # image becomes 12 x 12
        self.c4 = nn.Conv2d(in_channels= 64, out_channels= 64, kernel_size = 3) #image becomes 10 x 10
        self.max_pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2) # image becomes 5 x 5
        
        self.fc1 = nn.Linear(1600, 128) #first value seems to be total number of pixels of final image. Calculated by final number of channels * final image size 
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
