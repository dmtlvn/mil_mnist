import torch
import numpy as np

from functools import partial
from random import lognormvariate, random, randint
from torchvision.datasets import MNIST
from torchvision import transforms as T
from torch.utils.data import Dataset


class Dataset:
    
    def __init__(self, 
                 train: bool = True, 
                 target_class: int = 7, 
                 avg_size: float = 10, 
                 size_std: float = 2, 
                 size_range = (5, 250_000_000)
                ):
        """Samples image bags from MNIST. Bags contain variable number of images
        according to the specified mean-std. Log-normal distribution of bag sized
        is used.
        
        Parameters:
            train (bool) - train or test MNIST split. Default: True
            target_class (int) - "positive" digit class. Default: 7
            avg_size (float) - average bag size. Default: 10.0
            size_std (float) - bag size std. Default. 2.0
            size_range (tuple) - min-max bag size range. Default: (5, 250_000_000)
            
        Return:
            bag (torch.tensor) - [N,1,28,28]-dimensional tensor of images, 
                normalized to [-1, +1]
            target (int) - bag label 0/1
            idx (torch.tensor) - [N,]-dimensional tensor with indices of bag images 
                in the original MNIST dataset
        """
        
        self.target_class = target_class
        self.range = size_range
        
        # bag size distribution
        # designed to have fixed mean and std and heavier tail
        mu = np.log(avg_size**2 / np.sqrt(avg_size**2 + size_std**2))
        sigma = np.sqrt(np.log(1 + size_std**2 / avg_size**2))
        self.size_sampler = partial(lognormvariate, mu = mu, sigma = sigma)
        
        if train:
            transform = [
                T.RandomAffine(
                    degrees = 10, 
                    translate = (0.1, 0.1), 
                    scale = (0.9, 1.1),
                    interpolation = T.InterpolationMode.BICUBIC
                )
            ]
        else:
            # no test-time aug
            transform = []
            
        transform = transform + [T.ToTensor(), T.Normalize(mean = (0.5,), std = (0.5,))]
        transform = T.Compose(transform)
        
        self.mnist = MNIST(root = '../datasets/', train = train, transform = transform)
        self.positive_idx = torch.nonzero(self.mnist.targets == target_class).squeeze()
        self.negative_idx = torch.nonzero(self.mnist.targets != target_class).squeeze()
        
    def __len__(self):
        return 2**32 - 1
    
    def get_bag_size(self):
        size = round(self.size_sampler())
        size = min(max(size, self.range[0]), self.range[1])
        return int(size)
    
    def get_positive_sample(self):
        k = randint(0, self.positive_idx.shape[0] - 1)
        idx = self.positive_idx[k]
        img, label = self.mnist[idx]
        return img, idx
    
    def get_negative_sample(self):
        k = randint(0, self.negative_idx.shape[0] - 1)
        idx = self.negative_idx[k]
        img, label = self.mnist[idx]
        return img, idx
    
    def __getitem__(self, _):
        n = self.get_bag_size()
        
        bag = torch.zeros(n, 1, 28, 28).float()
        idx = torch.zeros(n).long()
        target = 0

        # stratified sampling
        p = 1 - 0.5**(1/n)
        for i in range(n):
            if random() < p:
                image, k = self.get_positive_sample()
                target = 1
            else:
                image, k = self.get_negative_sample()
            bag[i, 0] = image
            idx[i] = k
            
        return bag, target, idx
    