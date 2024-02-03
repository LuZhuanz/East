import os
import logging
import time



class logger:
    def __init__(self):
        self.time = 0

    def logger_init(self):
        self.time = time.time()
        # 创建 log 目录
        if not os.path.exists('logs'):
            os.makedirs('logs')

        # 配置日志记录
        logging.basicConfig(filename='logs/training.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        
    def logger_write_train(self, epoch, avg_train_loss, avg_val_loss, val_accuracy):
        now_time = time.time()
        spend = now_time - self.time
        self.time = now_time
        logging.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Time: {spend}s")
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Time: {spend}s")
        
    def logger_write_ckpt(self, save_path):
        logging.info(f"Model improved and saved to {save_path}")
        print(f"Model improved and saved to {save_path}")