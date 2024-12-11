import pandas as pd
import torch
from torch.utils.data import Dataset


class IMAGE(Dataset):
    def __init__(self, X, y, transform=None):
        # super(SERVER, self).__init__()
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # 需要根据self.X, self.y类型转化成张量
        X = torch.tensor(self.X.iloc[idx].values if isinstance(self.X, pd.DataFrame) else self.X[idx],
                         dtype=torch.float32).unsqueeze(0)  # 添加通道维度
        if X.dim() == 2:
            X = X.unsqueeze(-1)  # 形状从 [1, 17] 变为 [1, 17, 1]
        y = torch.tensor(self.y.iloc[idx] if isinstance(self.y, pd.Series) else self.y[idx], dtype=torch.float32)

        if self.transform:
            X = self.transform(X)

        return X, y
