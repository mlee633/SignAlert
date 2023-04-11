import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import math

class userData(Dataset):
    def __init__(self, file,transform =None):
        # data loading
        xy = np.genfromtxt(file, delimiter=",", dtype=np.uint8)[1:, :]
        self.x = (xy[:,1:])
        self.y = (xy[:,[0]]) #number of samples, 1
        self.n_samples = xy.shape[0]
        self.transform = transform
    def __getitem__(self,index):
        #first_image = self.x[index].reshape((28, 28))
        #im = Image.fromarray(first_image)
        #im = im.convert("L")
        image = self.x[index].reshape((28,28))
        if self.transform:
            return  self.transform(image), self.y[index]
        return image, self.y[index]
        
    def __len__(self):
        return self.n_samples