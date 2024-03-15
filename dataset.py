import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
from constant_test import *
from torchvision import datasets, transforms
import ast
from tqdm import tqdm

def data_loader():
    train_ratio = 0.8
    val_ratio = 0.2
    dataset = Mahjong_discard(txt_folder='data/discard')
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False,num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,num_workers=2)
    return train_loader, val_loader

def one_hot_mah(input):
    frequency = np.zeros(34, dtype=int)
    for i in range(0, 34):
        frequency[i] = np.count_nonzero(input == i)
    one_hot_matrix = np.zeros((4, 34), dtype=int)

    for index, freq in enumerate(frequency):
        if freq > 0:
            one_hot_matrix[:freq, index] = 1
            
    return one_hot_matrix

def one_hot_round(n):
    one_hot_matrix = np.zeros((4,34),dtype=int)
    #print(n)
    for i in range(2*n,2*n+4):
        one_hot_matrix[:,i-1] = 1
        
    return one_hot_matrix
        
def process_features(features, function=None):
    processed = []
    for feature in features:
        #print(feature)
        # 首先，将字符串表示的列表转换为实际的列表
        if not isinstance(feature, list):
            feature = string_to_int_list(feature)
        # 然后，将列表中的每个元素转换为整数，并执行地板除
        processed_feature = np.array(feature).astype(int) // 4
        processed.append(processed_feature) 
    return [function(feature) for feature in processed]


def string_to_int_list(string):
    # 使用ast.literal_eval安全地将字符串转换为列表
    return ast.literal_eval(string)

class Mahjong_discard(Dataset):
    def __init__(self, txt_folder):
        # 读取所有文本文件的路径
        self.file_paths = [os.path.join(txt_folder, fname) for fname in os.listdir(txt_folder) if fname.endswith('.txt')]
        # 每个文本文件有相同数量的行
        self.samples_per_file = 10000

    def __len__(self):
        # 总样本数是文件数乘以每个文件的样本数
        return len(self.file_paths) * self.samples_per_file

    def __getitem__(self, idx):
        # 确定文件索引和文件内的样本索引
        file_idx = idx // self.samples_per_file
        print(file_idx)
        sample_idx = idx % self.samples_per_file
        print(sample_idx)

        # 读取对应文件的特定行
        with open(self.file_paths[file_idx], 'r') as file:  #纯粹的cnn方法
            for i, line in enumerate(file):
                if i == sample_idx:
                    # 解析样本数据
                    data = line.split('$')
                    label = int(data[1])//4   #指示出哪一张牌
                    feature_0 = data[2:]
                    for i in [3, 6, 7, 2, 4, 5]:
                        feature_0[i] = string_to_int_list(feature_0[i])
                    
                    features_to_process = [
                        feature_0[3],  # hai_own
                        feature_0[6],  # meld_own
                        feature_0[7][0],  # meld_else_1
                        feature_0[7][1],  # meld_else_2
                        feature_0[7][2],  # meld_else_3
                        feature_0[2],  # dora
                        feature_0[4],  # discard_own
                        feature_0[4][:-1], 
                        feature_0[4][:-2],
                        feature_0[4][:-3],
                        feature_0[5][0],  # discard_1
                        feature_0[5][0][:-1],
                        feature_0[5][0][:-2],
                        feature_0[5][0][:-3],  
                        feature_0[5][1],  # discard_2
                        feature_0[5][1][:-1],
                        feature_0[5][1][:-2],
                        feature_0[5][1][:-3],
                        feature_0[5][2],  # discard_3
                        feature_0[5][2][:-1],
                        feature_0[5][2][:-2],
                        feature_0[5][2][:-3],
                        ]
                    processed_features = process_features(features_to_process, one_hot_mah)
                    feature_0[0] = int(feature_0[0])
                    round_ = one_hot_round(feature_0[0])
                    processed_features.append(round_)
                    combined_array = np.vstack(processed_features)
                    feature = torch.tensor(combined_array, dtype=torch.float32)
                    #print(feature)

                    return feature, label




#for debug
                """
transform = transforms.Compose([
    transforms.Resize((224, 224)),  
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
train_dataset_debug = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
val_dataset_debug = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

train_loader_ = DataLoader(train_dataset_debug, batch_size=batch_size, shuffle=False)
val_loader_ = DataLoader(val_dataset_debug, batch_size=batch_size, shuffle=False)

"""
error_indices = []
def check_dataloader_and_save_indices(dataloader):
    for i, data in enumerate(dataloader):
        try:
            # 尝试处理数据，比如通过模型
            outputs = data
            print(f"Processing batch {i+1}/{len(dataloader)}")
            #pass
        except Exception as e:
            print(f"Error processing batch {i+1}: {e}")
            error_indices.append(i)  # 保存出错的索引
    # 在迭代结束后保存出错的索引
    save_error_indices(error_indices)

def save_error_indices(indices):
    with open('error_indices.txt', 'w') as f:
        for index in indices:
            f.write(f"{index}\n")
    print("Saved error indices to 'error_indices.txt'.")

def is_valid_format(input_data):
    # 检查是否是元组
    if not isinstance(input_data, tuple):
        return False
    # 检查元组长度
    if len(input_data) != 2:
        return False
    # 检查第一个元素是否是torch.Tensor
    if not isinstance(input_data[0], torch.Tensor):
        return False
    # 检查第二个元素是否是int
    if not isinstance(input_data[1], int):
        return False
    return True
"""
dataloader, a = data_loader()
#check_dataloader_and_save_indices(dataloader)
Mahjong_discard = Mahjong_discard(txt_folder='data/discard')
for i in tqdm(range(95220000)):
    data = Mahjong_discard.__getitem__(i)
    if is_valid_format(data) == False:
        print(i, data)
        """