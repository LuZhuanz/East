import torch
import torch.nn as nn
import torch.nn.functional as F

from model_debug import *

def initialize_model(num_classes):
    model = East_cnn()
    return model

class ResNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = F.relu(out)
        return out

class East_cnn(nn.Module):
    def __init__(self):
        super(East_cnn, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(1,3), padding='same')
        self.res_blocks = nn.Sequential(*[ResNetBlock(128, 128, (1,3)) for _ in range(6)])
        self.fc = nn.Linear(92*34*128, 34)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = self.res_blocks(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x