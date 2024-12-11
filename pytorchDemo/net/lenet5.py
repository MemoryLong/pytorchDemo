import torch
import torch.nn as nn
import torch.nn.functional as f


# LeNet-5 是经典的卷积神经网络（CNN）架构之一，由 Yann LeCun 等人在 1998 年提出，用于手写数字识别任务（如 MNIST 数据集）。
# LeNet-5 是深度学习的早期突破性成果之一，其结构简单且高效，适合初学者理解卷积神经网络的基本工作原理。
# 示例使用 MNIST 数据集，输入图片为32×32
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        # self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积层 1 + 池化层 1
        x = f.relu(self.conv1(x))   # 激活
        x = f.max_pool2d(x, kernel_size=2, stride=2)  # 最大池化
        # 卷积层 2 + 池化层 2
        x = f.relu(self.conv2(x))
        x = f.max_pool2d(x, kernel_size=2, stride=2)  # 最大池化

        # 展平
        x = torch.flatten(x, 1)  # 展平成 (batch_size, 16*5*5)
        # 全连接层
        x = f.relu(self.fc1(x))
        x = f.relu(self.fc2(x))
        x = self.fc3(x)  # 输出 logits

        return x

