# ###############################################基础环境############################################
# Windows 10
# 显卡MX150
# cuda_11.0.3_451.82_win10
# cudnn-windows-x86_64-8.9.7.29_cuda11-archive
# python 3.8.19
# torch 1.7.1
# Microsoft Visual C++ 2015-2019 Redistributable (x64)
# #############################################组件安装#############################################
# conda install Pillow == 7.2
# torchvision手工安装whl文件
# conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=11.0 -c pytorch
# conda install tensorflow
# #############################################验证安装#############################################
# import torch
# print(torch.version.cuda)
# print(torch.backends.cudnn.version())
# print(torch.version.cuda)  # 输出 CUDA 版本
# print(torch.cuda.is_available())
# #################################################################################################
import os

from torchvision.utils import make_grid
from visdom import Visdom
import subprocess
import time

import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn

from loadData import load_data
from net.FullyConnected import FullyConnectedModel
from net.TextCNN import TextCNN

from net.lenet5 import LeNet5
from net.resnet18 import ResNet18
from net.demoNet import DemoNet
from net.LSTM import LSTMModel
from util.EarlyStopping import EarlyStopping


def pytorch_train():
    dnn = "ResNet18"
    dnn = "FULLY-CONNECTED"
    dnn = "LeNet5"
    dnn = "CNN"
    dnn = "LSTM"
    dataset_type = "MNIST"
    dataset_type = "CIFAR100"
    dataset_type = "TimeSeriesSERVER"
    dataset_type = "SERVER"

    lr = 0.001
    load_workers = 4
    num_epochs = 100  # 15
    patience = 5
    num_channels = 16
    num_classes = 2

    project_dir = "D:/Dream Future Project/workspace/pytorchDemo/"

    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打印当前选择的设备
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{current_time}] Using device: {torch.cuda.get_device_name(device)}")
    # 加载数据
    train_loader, val_loader, test_loader = load_data(load_workers, dataset_type)

    if dnn == "LeNet5":
        net = LeNet5().to(device)
    elif dnn == "ResNet18":
        net = ResNet18().to(device)
    elif dnn == "LSTM":
        # feature_size = train_loader.dataset.X.shape[1]
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            feature_size = len(inputs[0])  # 获取输入特征维度.feature_size
            break
        hidden_size = 32
        num_layers = 1
        num_classes = len(train_loader.dataset.y.unique())
        net = LSTMModel(feature_size, hidden_size, num_layers, num_classes).to(device)
    elif dnn == "FULLY-CONNECTED":
        # 模型参数
        # print(f"数据集的总样本数:{len(train_loader.dataset)}")  # 获取数据集的总样本数
        # print(f"数据集的总样本数:{train_loader.dataset.X.shape}")  # 获取数据集的总样本数
        # print(f"数据集的总样本数:{train_loader.dataset.y.shape}")  # 获取数据集的总样本数
        # print(f"数据集的总样本数:{train_loader[0][0]}")  # 获取数据集的总样本数
        # print(f"数据集的总样本数:{train_loader.dataset.y.unique()}")  # 获取数据集的总样本数
        # print(f"第一个批次的形状:{next(iter(train_loader))}")  # 获取第一个批次的形状
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            # print(f"Batch {batch_idx}")
            # print(f"Inputs Shape: {inputs.shape}")
            # print(f"Labels Shape: {labels.shape}")
            # input_size = len(inputs)  # 输入特征维度
            # print(f"特征维度{input_size}")
            feature_size = len(inputs[0])  # 获取输入特征维度.feature_size
            print(f"特征维度{feature_size}")
            break
        hidden_size = 32
        num_classes = len(train_loader.dataset.y.unique())
        net = FullyConnectedModel(feature_size, hidden_size, num_classes).to(device)
    elif dnn == "CNN":
        # vocab_size = len(vocab)
        embed_size = 50
        # kernel_sizes = [2, 3, 4]
        # # 模型、损失函数和优化器
        # net = TextCNN(vocab_size, embed_size, num_classes, kernel_sizes, num_channels).to(device)
    else:
        net = DemoNet().to(device)

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{current_time}] 神经网络模型: {dnn}")
    print(f"[{current_time}] 学习率Learning Rate: {lr}")
    print(f"[{current_time}] 训练迭代次数: {num_epochs}")
    print(f"[{current_time}] load_workers: {load_workers}")

    # TensorBoardX 可视化
    # writer = SummaryWriter(logdir='./logs/lenet5')  # 日志存储路径
    vis = setup_visdom(project_dir)
    # vis.close()

    # TODO 增量训练
    # 待识别存量模型特征匹配
    model_path = project_dir + "model/" + str.lower(dataset_type) + "-" + dnn + "-" + str(lr) + "-" + str(
        num_epochs) + ".model"
    try:
        # 创建模型实例并加载权重
        checkpoint = torch.load(model_path, map_location=device)
        # 加载训练好的模型权重
        # 使用 strict=False 来忽略缺失的键和不匹配的键
        pre_model = net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    except Exception as e:
        print(f"Error loading model: {e}")
        pre_model = None
    if pre_model is not None:
        # lr = lr * 0.5  # 新sample减少对预训练模型影响
        model_path = project_dir + "model/" + str.lower(
            dataset_type) + "-" + dnn + "-" + str(lr) + "-" + str(num_epochs * 2) + ".model"
        train_model(train_loader, val_loader, net, device, lr, num_epochs, model_path, patience, vis)
    else:
        # 单进程训练模型
        train_model(train_loader, val_loader, net, device, lr, num_epochs, model_path, patience, vis)
        # TODO:使用多进程训练模型

    # 单进程评估模型
    evaluate_model(test_loader, net, device, model_path, vis)
    # writer.close()

    torch.cuda.empty_cache()


def train_model(train_loader, val_loader, net, device, lr, num_epochs, model_path, patience, vis):
    early_stopping = EarlyStopping(patience=patience, save_path=model_path)
    # 损失函数：加权的交叉熵损失。
    # CrossEntropyLoss 是分类任务中的常用损失函数，通常不需要调整。
    # 如果遇到类别不平衡的问题，可以考虑使用加权的交叉熵损失函数。
    # weights = torch.tensor([0.1, 0.2, 0.7])  # 示例权重
    # criterion = nn.CrossEntropyLoss(weight=weights)
    criterion = nn.CrossEntropyLoss()

    # 优化器：SGD、Adam等
    # SGD 是一种基础的优化器，但在许多情况下，使用带有动量的 SGD 可能会收敛得更慢。
    # 坚持使用 SGD 优化器，可以尝试调整动量参数。通常，动量参数设置在 0.9 到 0.99 之间。
    # momentum = 0.9
    # print(f"动量momentum: {momentum}")
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    # Adam 通常在实践中表现得更好
    # Adam 会自适应调整每个参数的学习率，使得它对不同的损失函数、梯度规模更加鲁棒。
    # 引入正则化技术，防止模型过拟合。这会对权重施加一个惩罚项，鼓励模型保持更小的权重值。
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)

    # 学习率调度器
    # 每step_size个epoch，学习率会减少一次。
    # 每次减少学习率时，学习率会乘以gamma。
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    print_size = 1000 / 10
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f"[{current_time}] 每{print_size}批次输出一次")
    # 训练过程
    for epoch in range(num_epochs):  # 循环多次训练
        print_loss = 0.0  # 初始化一个变量，用于累计每个批次的损失值。
        train_loss = 0
        val_loss = 0
        net.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader, 0):
            inputs, labels = inputs.to(device), labels.to(device)  # 确保数据也在设备上

            # 向前传播
            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # 反向传播 + 优化
            optimizer.zero_grad()  # 清除之前的梯度信息，准备进行新一轮的反向传播
            loss.backward()  # 执行反向传播，计算w梯度。
            optimizer.step()  # 更新模型的参数（权重和偏置）。

            print_loss += loss.item()
            train_loss += loss.item()
            if batch_idx % print_size == print_size - 1:
                current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f'[{current_time}] Epoch [{epoch + 1}, {batch_idx + 1}] loss: {print_loss / print_size:.5f}')
                # writer.add_scalar('Loss/Train', print_loss / len(train_loader), epoch)
                # writer.add_scalar('Accuracy/Train', train_acc, epoch)

                # 显示训练图片
                disnum = min(100, inputs.size(0))  # 选取top N张图片拼接
                # img_grid = torch.cat([inputs[i] for i in range(disnum)], dim=2)
                img_grid = make_grid(inputs[:disnum], nrow=10, normalize=True, pad_value=1)
                vis.image(
                    # img_grid.squeeze(),
                    img_grid,
                    win="Images",
                    opts={"title": f"Epoch {epoch} Batch {batch_idx}: Training Images"}
                )

                # 显示预测结果
                predictions = torch.argmax(outputs, dim=1)[:disnum]  # 获取top N个预测值
                true_labels = labels[:disnum]
                vis.text(
                    "<br>".join(
                        [
                            f"Prediction: {pred}, True Label: {true}"
                            for pred, true in zip(predictions, true_labels)
                        ]
                    ),
                    win="Predictions",
                    # opts={"title": "Predictions vs True Labels"},
                    opts={"title": f"Epoch {epoch} Batch {batch_idx}: Training Images"}
                )

                vis.line(
                    X=[epoch * len(train_loader) + batch_idx],
                    Y=[loss.item()],
                    # Y=[print_loss / 100],
                    win="loss",
                    update="append",
                    name="Train Loss",
                    opts={"title": "Training Loss", "xlabel": "Iteration", "ylabel": "Loss"},
                )
                print_loss = 0.0

        # Validation phase
        net.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)  # 确保数据也在设备上
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(
            f"[{current_time}] Epoch {epoch + 1}, Train Loss Rate: {train_loss / len(train_loader):.4f}, Val Loss Rate: {val_loss / len(val_loader):.4f}")

        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        # Check early stopping condition
        early_stopping(val_loss, net)
        if early_stopping.early_stop:
            print(f"[{current_time}] Early stopping triggered.")
            break

        # # Load the best model
        # if os.path.exists(model_path):
        #     net.load_state_dict(torch.load(model_path))
        #     print("Loaded the best model.")

        # 每个 epoch 结束后，更新学习率
        scheduler.step()

    torch.save({
        'model_state_dict': net.state_dict(),
        # 'input_shape': input_shape,
        # 'optimizer_state_dict': optimizer.state_dict(),
        'epoch': num_epochs,
        'device': device,
        'lr': lr,
        # 'loss': loss,
    }, model_path)

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f'[{current_time}] Finished Training')


# 评估模型
def evaluate_model(test_loader, net, device, model_path, vis):
    checkpoint = torch.load(model_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()  # 设置模型为评估模式
    correct = 0
    total = 0
    # 禁用梯度计算以加速推理
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # writer.add_scalar('Accuracy/Test', 100 * correct / total)
            # vis.image(data.view(-1, 1, 28, 28), win='x')
            # vis.text(str(predicted.detach().cpu().numpy()), win='pred', opts=dict(title='pred'))

    acc = 100 * correct / total
    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    print(f'[{current_time}] Accuracy of the network on the {len(test_loader)} batch test images: {acc:.3f}%')
    vis.line(
        Y=[acc],
        X=[len(vis.get_window_data())],
        win="accuracy",
        update="append",
        name="Test Accuracy",
    )


# 可视化配置
def setup_visdom(project_dir):
    start_visdom(project_dir)
    vis = Visdom(use_incoming_socket=False)
    vis.line(
        Y=[0],
        X=[0],
        win="loss",
        opts=dict(title="Training Loss", xlabel="Batch", ylabel="Loss"),
        name="Train Loss",
    )
    vis.line(
        Y=[0],
        X=[0],
        win="accuracy",
        opts=dict(title="Test Accuracy", xlabel="Epoch", ylabel="Accuracy (%)"),
        name="Test Accuracy",
    )
    return vis


def start_visdom(project_dir):
    # 打开日志文件用于写入
    with open(project_dir + "logs/visdom.log", "w") as log:
        # 启动 Visdom 服务
        process = subprocess.Popen(
            ["python", "-m", "visdom.server"],  # 命令和参数
            stdout=log,  # 标准输出重定向到日志文件
            stderr=subprocess.STDOUT,  # 错误输出合并到标准输出
        )
        # 获取进程 ID 并保存到文件
        pid = process.pid
        # pid_file = "visdom.pid"
        # with open(pid_file, "w") as f:
        #     f.write(str(pid))
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"[{current_time}] Visdom server started with PID: {pid}")
        # 等待 Visdom 服务启动
        time.sleep(1)


if __name__ == '__main__':
    pytorch_train()
