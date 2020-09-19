import os
import numpy as np
import time
from PIL import Image
from torchvision import datasets,transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy
from glob import glob

def dataset_stats(path):
    files_training = glob(os.path.join(path,'Training', '*/*.jpg'))
    num_images = len(files_training)
    print('Number of images in Training file:', num_images)



    min_images = 1000
    im_cnt = []
    class_names = []
    print('{:18s}'.format('class'), end='')
    print('Count:')
    print('-' * 24)
    for folder in os.listdir(os.path.join(path, 'Training')):
        folder_num = len(os.listdir(os.path.join(path,'Training',folder)))
        im_cnt.append(folder_num)
        class_names.append(folder)
        print('{:20s}'.format(folder), end=' ')
        print(folder_num)
        if (folder_num < min_images):
            min_images = folder_num
            folder_name = folder
            
    num_classes = len(class_names)
    print("\nMinumum imgages per category:", min_images, 'Category:', folder)    
    print('Average number of Images per Category: {:.0f}'.format(np.array(im_cnt).mean()))
    print('Total number of classes: {}'.format(num_classes))
    return class_names, num_classes



class FruitTrainDataset(Dataset):
    def __init__(self, files, shuffle, split_val, class_names, transform=transforms.ToTensor()):
        self.shuffle = shuffle
        self.class_names = class_names
        self.split_val = split_val
        self.data = np.array([files[i] for i in shuffle[split_val:]])
        self.transform=transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        name = self.data[idx].split('/')[-2]
        y = self.class_names.index(name)
        img = self.transform(img)
            
        return img, y

class FruitValidDataset(Dataset):
    def __init__(self, files, shuffle, split_val, class_names, transform=transforms.ToTensor()):
        self.shuffle = shuffle
        self.class_names = class_names
        self.split_val = split_val
        self.data = np.array([files[i] for i in shuffle[:split_val]])
        self.transform=transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        name = self.data[idx].split('/')[-2]
        y = self.class_names.index(name)
        img = self.transform(img)
            
        return img, y
    
class FruitTestDataset(Dataset):
    def __init__(self, path, class_names, transform=transforms.ToTensor()):
        self.class_names = class_names
        self.data = np.array(glob(os.path.join(path, '*/*.jpg')))
        self.transform=transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img = Image.open(self.data[idx])
        name = self.data[idx].split('/')[-2]
        y = self.class_names.index(name)
        img = self.transform(img)
            
        return img, y
    
