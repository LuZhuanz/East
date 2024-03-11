from constant_test import *
from trainer import train_model
from dataset import train_loader_, val_loader_, data_loader
from model_debug import initialize_model
from utils import logger

def main():
    
    model = initialize_model(10).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(model)
    logger.write(model)
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    #train_loader, val_loader = data_loader()
    train_loader = train_loader_
    val_loader = val_loader_
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device='cuda', save_path = save_path)
    
    
if __name__ == "__main__":
    main()