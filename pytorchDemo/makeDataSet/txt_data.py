import numpy as np
import torch
from torch.utils.data import Dataset


class SERVER(Dataset):
    def __init__(self, X, y, transform=None):
        """
        :param self: 字典格式
        :param X: pandas格式
        :param y: pandas格式
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        feature_list = []
        label_list = []

        for col in self.X.columns:
            value = self.X.iloc[idx][col]
            feature_list = expand_by_line(idx, col, value, feature_list)
            # if isinstance(value, (int, float, np.integer, np.floating)):  # 数值数据直接添加
            #     feature_list.append(float(value))
            # elif isinstance(value, np.ndarray):  # 机器名字段已是数组
            #     feature_list.extend(value.tolist())  # 将数组展开为列表加入特征
            # else:
            #     raise ValueError(f"Unsupported data sample {idx} type in column {col}: {type(value)}")

        for col in self.y.columns:
            value = self.y.iloc[idx][col]
            label_list = expand_by_line(idx, col, value, label_list)
            # if isinstance(value, (int, float, np.integer, np.floating)):  # 数值数据直接添加
            #     label_list.append(float(value))
            # elif isinstance(value, np.ndarray):  # 机器名字段已是数组
            #     label_list.extend(value.tolist())  # 将数组展开为列表加入特征
            # else:
            #     raise ValueError(f"Unsupported data sample {idx} type in column {col}: {type(value)}")

        # 处理输入特征为张量
        X = torch.tensor(feature_list, dtype=torch.float32)
        # 标签处理为张量
        # y = torch.tensor(self.y.iloc[idx] if isinstance(self.y, pd.Series) else self.y[idx], dtype=torch.long)
        y = torch.tensor(label_list, dtype=torch.int64)

        return X, y


def expand_by_line(idx, col, value, value_list):
    if isinstance(value, (int, float, np.integer, np.floating)):  # 数值数据直接添加
        value_list.append(float(value))
    elif isinstance(value, np.ndarray):  # 机器名字段已是数组
        value_list.extend(value.tolist())  # 将数组展开为列表加入特征
    else:
        raise ValueError(f"Unsupported data sample {idx} type in column {col}: {type(value)}")

    return value_list
