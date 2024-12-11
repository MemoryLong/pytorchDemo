import torch.nn as nn


class FullyConnectedModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FullyConnectedModel, self).__init__()
        # Attention 层
        self.attention = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 动态生成特征权重
        attention_weights = self.attention(x)
        x = x * attention_weights  # 按元素加权
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
