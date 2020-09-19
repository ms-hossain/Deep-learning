import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import load_files
import torch.optim as optim
import os
import numpy as np
import time
from PIL import Image
from torchvision.utils import make_grid
from torchvision import datasets,transforms
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from model import Net
from dataloader import FruitTrainDataset, FruitValidDataset, FruitTestDataset, dataset_stats
from utils import imshow
from torch.autograd import Variable
import matplotlib.pyplot as plt
import copy
from glob import glob


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = '../datasets/fruits-360'
files_training = glob(os.path.join(path,'Training', '*/*.jpg'))
num_images = len(files_training)
print('Number of images in Training file:', num_images)

class_names, num_classes = dataset_stats(path)
tensor_transform = transforms.Compose([
    transforms.ToTensor()
])
all_data = ImageFolder(os.path.join(path, 'Training'), tensor_transform)
data_loader = torch.utils.data.DataLoader(all_data, batch_size=512, shuffle=True)
pop_mean = []
pop_std = []

for i, data in enumerate(data_loader, 0):
    numpy_image = data[0].numpy()
    
    batch_mean = np.mean(numpy_image, axis=(0,2,3))
    batch_std = np.std(numpy_image, axis=(0,2,3))
    
    pop_mean.append(batch_mean)
    pop_std.append(batch_std)

pop_mean = np.array(pop_mean).mean(axis=0)
pop_std = np.array(pop_std).mean(axis=0)
print(pop_mean)
print(pop_std)
np.random.seed(123)
shuffle = np.random.permutation(num_images)

split_val = int(num_images * 0.2)
print('Total number of images:', num_images)
print('Number of valid images after split:',len(shuffle[:split_val]))
print('Number of train images after split:',len(shuffle[split_val:]))

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(pop_mean, pop_std) # These were the mean and standard deviations that we calculated earlier.
    ]),
    'Test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(pop_mean, pop_std) # These were the mean and standard deviations that we calculated earlier.
    ]),
    'valid': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(pop_mean, pop_std) # These were the mean and standard deviations that we calculated earlier.
    ])
}

train_dataset = FruitTrainDataset(files_training, shuffle, split_val, class_names, data_transforms['train'])
valid_dataset = FruitValidDataset(files_training, shuffle, split_val, class_names, data_transforms['valid'])
test_dataset = FruitTestDataset('../datasets/fruits-360/Test', class_names, transform=data_transforms['Test'])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=16, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=True)

# Here we are just checking out the next batch of images from the train_loader, and below I print the class names. 
inputs, classes = next(iter(train_loader))
out = make_grid(inputs)

cats = ['' for x in range(len(classes))]
for i in range(len(classes)):
    cats[i] = class_names[classes[i].item()]
    
imshow(out, pop_std, pop_mean, 'Grid View')
# print(cats)

### These are just used for convinience purposes

dataloaders = {
    'train': train_loader,
    'valid': valid_loader,
    'Test': test_loader
}
dataset_sizes = {
    'train': len(train_dataset),
    'valid': len(valid_dataset),
    'Test': len(test_dataset)
}

model = Net(num_classes)
model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
exp_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)


def fit(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time() # allows us to keep track of how long it took
    
    best_acc = 0.0 # allows us to store the best_acc rate (for validation stage)
    
    # Loop through the data-set num_epochs times.
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 15)
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train() # This sets the model to training mode
            else:
                model.eval() # this sets the model to evaluation mode 
                
            running_loss = 0.0
            running_corrects = 0
            
            # using the dataloaders to load data in batches
            for inputs, labels in dataloaders[phase]:
                # putting the inputs and labels on cuda (gpu)
                inputs = inputs.to(device) 
                labels = labels.to(device)
                
                # zero the gradient
                optimizer.zero_grad()
                
                # if training phase, allow calculating the gradient, but don't allow otherwise
                with torch.set_grad_enabled(phase == 'train'):
                    # get outputs and predictions
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    
                    loss = criterion(outputs, labels) # get value of loss function with the current weights 
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            # keep track of the best weights for the validation dataset 
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                
        print()
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best validation Acc: {:4f}'.format(best_acc))
    
    model.load_state_dict(best_model_wts)
    return model

model = fit(model, criterion, optimizer, exp_scheduler, num_epochs=30)

correct = 0
total = 0
with torch.no_grad():
    for images, labels in dataloaders['Test']:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the {} test images: {:.3f}%'.format(dataset_sizes['Test'], 
                                                                      100 * correct / dataset_sizes['Test']))