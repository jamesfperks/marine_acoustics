"""
Define CNN model using pytorch.

"""


import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryNet(nn.Module):
    def __init__(self):
        super(BinaryNet, self).__init__()
        # Convoultional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(1024, 120)
        self.fc1_drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(120, 84)
        self.fc2_drop = nn.Dropout(p=0.5)
        
        # Output layer
        self.fc3 = nn.Linear(84, 1)
        self.sigmoid =  nn.Sigmoid()

    def forward(self, x):
        # Convolutional layers
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        # Fully connected layers
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1_drop(self.fc1(x)))
        x = F.relu(self.fc1_drop(self.fc2(x)))
        
        # Output layer
        x = self.fc3(x)
        x = self.sigmoid(x)
        
        return x


class MultiNet(nn.Module):
    def __init__(self):
        super(MultiNet, self).__init__()
        # Convoultional layers
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256, 120)
        self.fc1_drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(120, 84)
        self.fc2_drop = nn.Dropout(p=0.5)
        
        # Output layer
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        
        # Fully connected layers
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1_drop(self.fc1(x)))
        x = F.relu(self.fc1_drop(self.fc2(x)))
        
        # Output layer
        x = self.fc3(x)
        
        return x