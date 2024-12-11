import time

import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split

from makeDataSet.txt_data import SERVER


def load_data(num_workers, dataset_type):
    train_set = None
    val_set = None
    test_set = None
    batch_size = 64
    val_ratio = 0.1  # 90% 训练集，10% 验证集
    project_dir = "D:/Dream Future Project/workspace/pytorchDemo/"

    if dataset_type == "MNIST":
        # 数据准备 数据预处理 Converts a PIL.Image or numpy.ndarray to torch.FloatTensor of shape (C x H x W)
        # and normalize in the range [0.0, 1.0]
        transform = transforms.Compose([
            # 数据增强（Data Augmentation）
            # 通过随机旋转、缩放、裁剪等方式对数据进行增强，可以有效提升模型的泛化能力。
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),   # 实际使用中导致准确率下降
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        # 下载训练数据集
        # https://yann.lecun.com/exdb/mnist/拒绝下载，需要手工找资源
        train_set = torchvision.datasets.MNIST(root=project_dir + 'dataset/',
                                               train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=project_dir + 'dataset/',
                                              train=False, download=True, transform=transform)
        val_size = int(len(train_set) * val_ratio)  # 验证集大小
        train_size = len(train_set) - val_size  # 剩余部分作为新的训练集
        # 随机拆分数据集
        train_set, val_set = random_split(train_set, [train_size, val_size])
    elif dataset_type == "CIFAR100":
        # 定义数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化
        ])
        # 下载训练数据集
        train_set = torchvision.datasets.CIFAR100(root=project_dir + 'dataset/CIFAR/', train=True, download=True,
                                                  transform=transform)
        test_set = torchvision.datasets.CIFAR100(root=project_dir + 'dataset/CIFAR/', train=False, download=True,
                                                 transform=transform)
        val_size = int(len(train_set) * val_ratio)  # 验证集大小
        train_size = len(train_set) - val_size  # 剩余部分作为新的训练集
        # 随机拆分数据集
        train_set, val_set = random_split(train_set, [train_size, val_size])
    elif dataset_type == "SERVER":
        batch_size = 8
        # 定义数据预处理
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Adjust image size (if needed)
            transforms.Grayscale(num_output_channels=1),  # 确保单通道
            transforms.ToTensor(),  # 转换为 PyTorch 张量
            transforms.Normalize((0.5,), (0.5,))  # 归一化
        ])

        # 加载数据集
        train_pt = torch.load(project_dir + 'dataset/SERVER/train_dataset.pt')
        test_pt = torch.load(project_dir + 'dataset/SERVER/test_dataset.pt')

        # 使用 train_test_split 拆分train数据集为train、val
        train_X, val_X, train_y, val_y = train_test_split(
            train_pt['X'], train_pt['y'], test_size=val_ratio, random_state=42)

        print(f"加载feature shape：{train_X.shape}")
        print(f"feature数据：共{len(train_X)}行\t共{len(train_X.columns)}列")
        for col in train_y.columns:
            print(f"feature列名：{train_y[col].name}\t列类型：{train_y[col].dtype}")

        # Pass transform when creating the dataset instance
        # Assuming the loaded data contains 'X' and 'y'
        test_set = SERVER(test_pt['X'], test_pt['y'], transform=transform)

        # 将数据重新封装为 DataFrame
        train_X = pd.DataFrame(train_X)
        train_y = pd.DataFrame(train_y)
        val_X = pd.DataFrame(val_X)
        val_y = pd.DataFrame(val_y)

        train_set = TensorDataset(train_X, train_y)
        val_set = TensorDataset(val_X, val_y)

    elif dataset_type == "TimeSeriesSERVER":
        batch_size = 2
        pass
    else:
        pass

    # 加载数据集
    # shuffle：是否在每个 epoch（训练周期）之前对数据进行随机洗牌。如果设置为 True，数据将在每个 epoch 开始时重新洗牌，以增加训练的随机性。
    # num_workers：用于数据加载的并行工作线程数。
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{current_time}] 训练数据集长度：{format(len(train_set))}")
    print(f"[{current_time}] 测试数据集长度: {format(len(test_set))}")
    print(f"[{current_time}] batchSize: {batch_size}")

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    load_data(num_workers=2, dataset_type="SERVER")
