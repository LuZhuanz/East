import torch
from torch.utils.data import Dataset
import os

class Mahjong_discard(Dataset):
    def __init__(self, txt_folder):
        # 读取所有文本文件的路径
        self.file_paths = [os.path.join(txt_folder, fname) for fname in os.listdir(txt_folder) if fname.endswith('.txt')]
        # 假设每个文本文件有相同数量的行
        self.samples_per_file = 10000

    def __len__(self):
        # 总样本数是文件数乘以每个文件的样本数
        return len(self.file_paths) * self.samples_per_file

    def __getitem__(self, idx):
        # 确定文件索引和文件内的样本索引
        file_idx = idx // self.samples_per_file
        sample_idx = idx % self.samples_per_file

        # 读取对应文件的特定行
        with open(self.file_paths[file_idx], 'r') as file:
            for i, line in enumerate(file):
                if i == sample_idx:
                    # 解析样本数据
                    data = line.strip().split(',')
                    # 转换数据为所需的格式，例如转换为张量
                    data = torch.tensor([float(x) for x in data])
                    return data

# 使用自定义数据集
dataset = Mahjong_discard(txt_folder='discard')

# 创建 DataLoader
from torch.utils.data import DataLoader