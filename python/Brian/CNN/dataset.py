from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import math


class userData(Dataset):
    def __init__(self, array, transform =None):
        # data loading
        self.x = array[:, 1:]
        self.y = array[:, [0]]
        self.n_samples = array.shape[0]
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
    