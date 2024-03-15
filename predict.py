import torch
import numpy as np
import dataset

def predict(model, device, input_data, model_path):
    """
    加载模型并对输入数据进行推理。

    参数:
    model (torch.nn.Module): PyTorch模型的实例。
    device (torch.device): 运行模型的设备，如'cuda'或'cpu'。
    input_data (np.ndarray): 输入数据，应该与模型训练时的数据格式相匹配。
    model_path (str): 训练好的模型权重文件的路径。

    返回:
    np.ndarray: 模型的预测结果。
    """
    # 确保模型在正确的设备上
    model.to(device)

    # 加载模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))

    # 设置为评估模式
    model.eval()

    # 转换输入数据为torch.Tensor
    input_tensor = input_data.to(device)

    # 添加一个批次维度，如果它还没有
    
    input_tensor = input_tensor.unsqueeze(0)

    with torch.no_grad():  # 不计算梯度，以加速推理过程
        output = model(input_tensor)
    
    # 将输出转换回numpy数组
    return output.cpu().numpy()

def main_predict(model, model_path):
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    while True:  # 开始循环
        # 获取用户输入
        print("请输入（输入'q'退出）：")
        user_input = input()  # 直接接收字符串输入
        if user_input == 'q':  # 如果用户输入'q'，退出循环
            break

        try:
            user_input = int(user_input)  # 尝试将输入转换为整数
        except ValueError:
            print("请输入有效的数字或'q'退出。")
            continue

        getitem = dataset.Mahjong_discard(txt_folder='data/discard')
        input_data, x = getitem.__getitem__(user_input)

        # 进行预测
        predictions = predict(model, device, input_data, model_path)
        pred = predictions.argmax()

        # 打印预测结果
        print("predict:", predictions, pred, "true:", x)

# 确保调用main_predict时传入正确的模型和模型路径参数
# main_predict(你的模型, 模型路径)