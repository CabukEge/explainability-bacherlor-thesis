
from sklearn.base import accuracy_score
from sklearn.tree import DecisionTreeClassifier
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

# Einfaches Netz
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3 * 3, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = x.view(-1, 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Convolutional Neural Network
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size=2, stride=1)
        self.fc1 = nn.Linear(4 * 2 * 2, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(-1, 4 * 2 * 2)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
