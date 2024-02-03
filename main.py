from constant_test import *
from trainer import train_model
from dataset import train_loader_, val_loader_
from model import initialize_model

def main():
    
    model = initialize_model(10).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(model)
    train_model(model, train_loader_, val_loader_, criterion, optimizer, scheduler, epochs, device='cuda', save_path = save_path)
    
    
if __name__ == "__main__":
    main()