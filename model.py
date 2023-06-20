import torch
from torch import nn


def init_weights(m):
    if isinstance(m, torch.nn.Linear) or isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)


class ConvBlock(torch.nn.Sequential):
    
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        super().__init__(
            torch.nn.Conv2d(in_channels, out_channels, *args, **kwargs),
            torch.nn.BatchNorm2d(num_features = out_channels),
            torch.nn.ReLU(),
        )


class AttentionPool(nn.Module):
    
    def __init__(self, num_features: int, inner_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features = num_features, out_features = inner_dim),
            nn.Tanh(),
            nn.Linear(in_features = inner_dim, out_features = 1),
        )
        
    def forward(self, bag):
        """
        Parameters:
            bag (torch.tensor) - [N, C]-dimensional tensor of image embeddings
            
        Returns:
            [1, C]-dimensional torch tensor
        """
        w = self.model(bag).softmax(dim = 0)
        return torch.mean(w * bag, dim = 0, keepdim = True)
    
    
class GatedAttentionPool(nn.Module):
    
    def __init__(self, num_features: int, inner_dim: int):
        super().__init__()
        self.att = nn.Sequential(
            nn.Linear(in_features = num_features, out_features = inner_dim),
            nn.Tanh(),
        )
        self.gate = nn.Sequential(
            nn.Linear(in_features = num_features, out_features = inner_dim),
            nn.Sigmoid(),
        )
        self.weight = nn.Linear(in_features = inner_dim, out_features = 1)
        
    def forward(self, bag):
        """
        Parameters:
            bag (torch.tensor) - [N, C]-dimensional tensor of image embeddings
            
        Returns:
            [1, C]-dimensional torch tensor
        """
        w = self.weight(self.att(bag) * self.gate(bag)).softmax(dim = 0)
        return torch.mean(w * bag, dim = 0)
    
    
class MILModel(nn.Module):
    
    def __init__(self, janossy_samples: int = 4, width: int = 1):
        """
        Parameters:
            janossy_samples (int) - number of bag permutations for Janossy pooling[1]. 
                Default: 4
            width (int) - network width scaling parameter. Default: 1
          
        References:
            [1] Janossy Pooling, 2019: https://arxiv.org/abs/1811.01900
        """
        super().__init__()
        self.janossy_samples = janossy_samples
        self.backbone = nn.Sequential(
            ConvBlock(in_channels = 1, out_channels = 16*width, kernel_size = 3, bias = False),
            nn.MaxPool2d(kernel_size = 2),
            ConvBlock(in_channels = 16*width, out_channels = 32*width, kernel_size = 3, bias = False),
            nn.MaxPool2d(kernel_size = 2),
            ConvBlock(in_channels = 32*width, out_channels = 64*width, kernel_size = 3, bias = False),
            nn.AvgPool2d(kernel_size = 3),
            nn.Conv2d(in_channels = 64*width, out_channels = 128*width, kernel_size = 1, bias = False),
            nn.BatchNorm2d(num_features = 128*width, affine = False)
        )
        self.bag_head = nn.Sequential(
            nn.Linear(in_features = 128*width, out_features = 256*width),
            nn.ReLU(),
            nn.Linear(in_features = 256*width, out_features = 1),
            nn.Sigmoid()
        )
        # to get instance-level predictions out of the box
        self.instance_head = nn.Sequential(
            nn.Linear(in_features = 128*width, out_features = 256*width),
            nn.ReLU(),
            nn.Linear(in_features = 256*width, out_features = 1),
            nn.Sigmoid()
        )
        self.pool = AttentionPool(num_features = 128*width, inner_dim = 256*width)
        self.apply(init_weights)
        
    def forward(self, x):
        """
        Parameters:
            bag (torch.tensor) - [N,1,28,28]-dimensional tensor. Image normalization 
                is to [-1, +1].
                
        Returns:
            embeddings (torch.tensor) - [N, 128*width]-dimensional tensor.
            bag_prob (torch.tensor) - bag-level predicted probability
            instance_probs (torch.tensor) - [N,]-dimensional tensor of instance-level
                predicted probabilities
        """
        embeddings = self.backbone(x).squeeze(3).squeeze(2)

        bag_feature = 0
        for i in range(self.janossy_samples):
            perm = torch.randperm(embeddings.shape[0])       
            bag_feature += self.pool(embeddings[perm]) / self.janossy_samples
            
        bag_prob = self.bag_head(bag_feature).squeeze()       
        instance_probs = self.instance_head(embeddings.detach()).squeeze()
        return embeddings, bag_prob, instance_probs
