import torch
from torch import nn


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_size, num_classes, kernel_sizes, num_channels):
        """
        :param vocab_size: 词汇表大小
        :param embed_size: 嵌入向量维度
        :param num_classes: 分类数
        :param kernel_sizes: 卷积核尺寸列表
        :param num_channels: 每种卷积核的输出通道数
        """
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_size, num_channels, kernel_size) for kernel_size in kernel_sizes
        ])
        self.fc = nn.Linear(num_channels * len(kernel_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)  # [batch_size, seq_len, embed_size]
        x = x.permute(0, 2, 1)  # 转换为 [batch_size, embed_size, seq_len]
        conv_outs = [torch.relu(conv(x)).max(dim=2).values for conv in self.convs]  # 每个核的 max pooling
        x = torch.cat(conv_outs, dim=1)  # 拼接不同卷积核的输出
        x = self.fc(x)  # 全连接层
        return x
