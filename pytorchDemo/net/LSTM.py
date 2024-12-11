import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.feature_weights = nn.Parameter(torch.ones(input_size))  # LSTM 前添加一个线性层，动态调整权重；初始化权重为 1
        # 线性层调整特征权重
        # self.feature_weights = nn.Linear(input_size, input_size)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # Attention 机制
        self.attention = nn.Linear(hidden_size, 1)
        # # 为了增强 Attention 的表达能力，可以引入一个双层的注意力模块
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )
        self.fc = nn.Linear(hidden_size, num_classes)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # # 初始化权重
        # self.init_weights()

    def forward(self, x):
        # 对特征加权
        x = x * self.feature_weights  # 按权重加权
        # 将时间步设为 1
        x = x.unsqueeze(1)  # 扩展时间维度：[batch_size, seq_len=1, input_size]
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        # LSTM 部分
        out, _ = self.lstm(x, (h0, c0))
        # out = self.fc(out[:, -1, :])  # 取最后一个时间步的输出
        # Attention 部分
        attention_weights = torch.softmax(self.attention(out), dim=1)
        context_vector = torch.sum(attention_weights * out, dim=1)
        # 输出层
        out = self.fc(context_vector)

        return out

    def init_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)
