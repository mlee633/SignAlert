
import torch.nn as nn

### Followed guide on making LeNet5 models: https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/


class LeNet5(nn.Module):
    def __init__(self, num_classes):
       
        super().__init__()
         ## Initialising the 2 layers we are going to use in the LeNet5 model
        self.layer1 = nn.Sequential(
            # Convolution with output having stide 1 and no padding
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
        # Relu is used for filtering
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes) # Getting num_classes of output (i.e 26 outputs)
     # Sequence in which the layers will process the image
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