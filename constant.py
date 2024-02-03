import torch
import torch.nn as nn
import torch.optim as optim
from model import model
#EAST parameters

#dataloader
#folder_path = 'data/xml2017'
folder_path = 'data/test'
batch_size = 64
shuffle=True
num_workers=4


#trainer
scheduler = None
epochs = 10
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

save_path='checkpoints/model_debug.pth'