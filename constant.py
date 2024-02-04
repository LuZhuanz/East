import torch
import torch.nn as nn
import torch.optim as optim

#EAST parameters

#dataloader
#folder_path = 'data/xml2017'
folder_path = 'data/discard'
batch_size = 64
shuffle=True
num_workers=4


#trainer
scheduler = None
epochs = 200
criterion = nn.CrossEntropyLoss()


save_path='checkpoints/model.pth'