import os
import logging


class logger:

    def logger_init():
        # 创建 log 目录
        if not os.path.exists('log'):
            os.makedirs('log')

        # 配置日志记录
        logging.basicConfig(filename='logs/training.log', level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
        
    def logger_write_train(epoch, avg_train_loss, avg_val_loss, val_accuracy):
        logging.info(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
    def logger_write_ckpt(save_path):
        logging.info(f"Model improved and saved to {save_path}")
        print(f"Model improved and saved to {save_path}")