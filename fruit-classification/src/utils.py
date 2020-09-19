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


# This allows us to see some of the fruits in each of the datasets 
def imshow(inp, pop_std, pop_mean, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = pop_std * inp + pop_mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    # plt.img_save('../output/grid_img.jpg')
    plt.savefig('{}/{}.jpg'.format('../output', 'grid_img'), format='jpg')
    plt.pause(0.001)  # pause a bit so that plots are updated

