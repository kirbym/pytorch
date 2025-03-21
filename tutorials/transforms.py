import torch
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

ds = datasets.FashionMNIST(
    root="../data",
    train=True,
    download=True,
    transform=ToTensor(),
    # Lambda transforms apply any user-defined lambda function. Here, we define a function 
    # to turn the integer into a one-hot encoded tensor. It first creates a zero tensor 
    # of size 10 (the number of labels in our dataset) and calls scatter_ which assigns a 
    # value=1 on the index as given by the label y.
    target_transform=Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
)