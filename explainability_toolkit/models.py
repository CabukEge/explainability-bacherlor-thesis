import torch
import torch.nn as nn



class SimpleNet(nn.Module):
    """
    Simple fully connected neural network for binary classification.
    Input: 9 features (3x3 grid)
    Output: 2 classes
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(3 * 3, 10)
        self.fc2 = nn.Linear(10, 2)

    def forward(self, x):
        x = x.view(-1, 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class ImprovedCNN(nn.Module):
    """
    Improved CNN architecture with batch normalization and dropout.
    Input: 1 channel 3x3 grid
    Output: 2 classes
    """
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 16, kernel_size=2, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 3 * 3, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        
        # Flatten and fully connected layers
        x = x.view(-1, 32 * 3 * 3)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x