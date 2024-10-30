import tensorflow as tf
from tensorflow import keras

# 加载Keras模型
model = keras.models.load_model("D:\\Study_Date\\DeepLearning-Emotion-Classifier-withGUI\\weight.h5")

# 将TensorFlow模型的权重转换为PyTorch模型
import torch
import torch.nn as nn

# 定义相应的PyTorch模型结构
class MyPyTorchModel(nn.Module):
    def __init__(self):
        super(MyPyTorchModel, self).__init__()
        # 添加层定义

    def forward(self, x):
        # 定义前向传播
        return x

pytorch_model = MyPyTorchModel()

# 转换权重
for layer in model.layers:
    # 根据需要将层的权重转换为PyTorch格式
    # 这里需要根据你的具体层结构进行调整
    if isinstance(layer, keras.layers.Dense):
        pytorch_layer = pytorch_model.fc[layer.name]  # 假设你的PyTorch模型有相应的层
        pytorch_layer.weight.data = torch.Tensor(layer.get_weights()[0]).t()  # 转置权重
        pytorch_layer.bias.data = torch.Tensor(layer.get_weights()[1])

# 保存PyTorch模型
torch.save(pytorch_model.state_dict(), "D:\\Study_Date\\DeepLearning-Emotion-Classifier-withGUI\\pytorch_weight.pth")
