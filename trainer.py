import torch
import torch.nn as nn
import torch.optim as optim
import os
import utils


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs, device, save_path):
    model.to(device)
    best_val_loss = float('inf')
    utils.logger.logger_init()

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss, val_accuracy = validate(model, val_loader, criterion, device)

        # 学习率调度
        if scheduler is not None:
            scheduler.step(avg_val_loss)

        # 日志记录
        utils.logger.logger_write_train(epoch, avg_train_loss, avg_val_loss, val_accuracy)

        # 保存模型（如果在验证集上的表现更好）
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            utils.logger.logger_write_ckpt(save_path)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_val_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_accuracy = 100. * correct / len(val_loader.dataset)
    return avg_val_loss, val_accuracy


