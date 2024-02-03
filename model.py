import torch
import torch.nn as nn
import torch.nn.functional as F

from model_debug import *

def initialize_model(num_classes):
    model = SimpleCNN(num_classes=num_classes)
    return model