from constant_test import *
from trainer import train_model
import predict
import dataset
from model import initialize_model
from utils import logger

def main(type):
    
    model = initialize_model(10).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(model)
    #logger.write(model)
    if type == 'train':
        optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        train_loader, val_loader = dataset.data_loader()
    #train_loader = train_loader_
    #val_loader = val_loader_
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device='cuda', save_path = save_path)
    elif type == 'predict':
        predict.main_predict(model,model_path="checkpoints/model.pth")
        
    else:
        print('ERROR')
    
if __name__ == "__main__":
    main(type='predict')