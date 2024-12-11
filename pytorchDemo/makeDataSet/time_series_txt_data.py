import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils.data import Dataset


class SERVER(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data_list = []
        for col in self.X.columns:
            value = self.X.iloc[idx][col]
            if isinstance(value, (int, float, np.integer, np.floating)):  # 数值数据直接添加
                data_list.append(float(value))
            elif isinstance(value, np.ndarray):  # 机器名字段已是数组
                data_list.extend(value.tolist())  # 将数组展开为列表加入特征
            else:
                raise ValueError(f"Unsupported data sample {idx} type in column {col}: {type(value)}")

        # 处理输入特征为张量
        X = torch.tensor(data_list, dtype=torch.float32)
        # 标签处理为张量
        y = torch.tensor(self.y.iloc[idx] if isinstance(self.y, pd.Series) else self.y[idx], dtype=torch.long)

        # print(f"type(self.X){type(self.X)}\tshape(X.shape){X.shape}\t")  # X.keys()\t{X.keys()}
        # print(f"type(self.y){type(self.y)}\tshape(y.shape){type(y.shape)}\t")

        return X, y
