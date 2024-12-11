import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences


def make_server_data():
    # 1. 数据加载
    project_dir = "D:/Dream Future Project/workspace/pytorchDemo/"
    encoder_dir = project_dir + "dataset/SERVER/"
    file_path = project_dir + "dataset/SERVER/server_list.csv"
    data = pd.read_csv(file_path)
    print(f"原始表格式：shape {data.shape}")
    print(f"原始表格式：共{len(data)}行\t共{len(data.columns)}列")
    for col in data.columns:
        print(f"原始列名：{data[col].name}\t列类型：{data[col].dtype}")

    # 2. 格式化表结构
    # 数据分组
    data['group'] = data['机器名']
    data.drop(columns=['机器名'], inplace=True)
    # data = data.sort_values(by='timestamp').groupby('group')

    # 时间特征处理
    data['timestamp'] = pd.to_datetime(data['创建/变更时间']).view('int64')
    # data['year'] = data['时间'].dt.year
    # data['month'] = data['时间'].dt.month
    # data['day'] = data['时间'].dt.day
    data.drop(columns=['创建/变更时间'], inplace=True)
    # TODO 初步评估随机日期导致命中率低均值化
    data.drop(columns=['timestamp'], inplace=True)

    # 加载数据解码器
    encoder_path = os.path.join(encoder_dir, f'encoder.pkl')
    encoders = init_encoders(encoder_path, data.columns)

    # 3. 数据清洗
    # 删除无关字段
    # data.drop(['机器名', '创建/变更时间'], axis=1, inplace=True)
    # 填充空值
    # data.fillna(data.median(), inplace=True)

    # # Label 编码
    # label_columns = 'label'
    # data[label_columns] = data[label_columns].str.split('-').str[0].astype(int)
    # data[label_columns] = LabelEncoder().fit_transform(data[label_columns])

    categorical_columns = ['label', '归属系统', '系统级别', '规格', '机房', '网络区域', '芯片架构']
    # 类别字段 one-hot 编码
    # data = pd.get_dummies(data, columns=categorical_columns)
    # 类别字段 Label 编码
    for col in categorical_columns:
        # 获取对应字段的编码器
        encoder = encoders[col]
        # 动态扩展编码器的类别
        new_categories = set(data[col]) - set(encoder.classes_)
        if new_categories:
            encoder.classes_ = list(encoder.classes_) + list(new_categories)
        encoders[col] = encoder

        # 对增量数据进行编码
        data[col] = encoder.fit_transform(data[col])

    # 文本序列化
    # TODO 词库/分词
    # TODO 文本语义相似性
    # TODO 文本大小写
    text_columns = ['group', "归属微服务", '进程', '业务IP']
    for col in text_columns:
        # 获取对应字段的编码器
        encoder = encoders[col]

        # 步骤1：拆分字符串
        data[col] = data[col].apply(lambda x: re.split('[- _;,.|/\t\n]', x))
        # 步骤2：序列化（LabelEncoder）
        # 需要将拆分后的每个元素序列化为数字
        flattened = [item for sublist in data[col] for item in sublist]  # 将所有词汇展开
        encoder.fit(flattened)
        encoded = data[col].apply(lambda x: encoder.transform(x).tolist())  # 对每行进行序列化

        # 动态扩展编码器的类别
        new_categories = set(flattened) - set(encoder.classes_)
        if new_categories:
            encoder.classes_ = list(encoder.classes_) + list(new_categories)
        encoders[col] = encoder

        # 步骤3：对齐序列长度（padding）
        max_len = max(encoded.apply(len))  # 获取最大长度
        padded_sequences = pad_sequences(encoded, padding='post', maxlen=max_len)
        data[col] = list(padded_sequences)

    # text_columns = ['机器名', '进程']
    # for col in text_columns:
    #     # 不关心字符串中不同部分（如连字符、数字等）的权重，只关心它们是否存在，CountVectorizer 会更适合
    #     # vectorizer = CountVectorizer(max_features=10)  # 只保留一个特征
    #     # 仅考虑词频（TF），还会考虑词的逆文档频率（IDF），用以减少在多个文档中频繁出现的词的权重，强调那些在少数文档中出现的词
    #     vectorizer = TfidfVectorizer(max_features=10)  # 只保留最高 TF-IDF 的特征
    #     sparse_matrix = vectorizer.fit_transform(data[col])  # 使用 fit_transform 获取稀疏矩阵
    #     dense_matrix = sparse_matrix.toarray()  # 转换为稠密矩阵（numpy 数组）
    #     # dense_matrix = [''.join(map(str, row)) for row in dense_matrix]
    #     data[col] = list(dense_matrix)  # 将稠密矩阵赋值回 DataFrame

    # # IP字段转数值
    # ip_columns = '业务IP'
    # data[ip_columns] = data[ip_columns].str.split('.')
    # data[ip_columns] = data[ip_columns].str[0].astype(int) * 256 * 256 * 256 + data[ip_columns].str[1].astype(
    #     int) * 256 * 256 + data[ip_columns].str[2].astype(int) * 256 + data[ip_columns].str[3].astype(int)

    # 数值字段归一化
    # TODO 存量、增量数据归一化
    numeric_columns = ['最高CPU利用率', '最高内存利用率', 'CPU利用率(>20%)时间占比', 'CPU利用率(>30%)时间占比']
    for col in numeric_columns:
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        # data[col] = data[col].values.reshape(-1, 1)  # 将一维数组 reshape 成二维数组
        data[col] = scaler.fit_transform(data[col].values.reshape(-1, 1))
        data[col] = data[col].astype(float)  # 转换为浮点型

    # non_feature_columns = ['时间', '机器名', 'label']  # 明确非特征列
    # sequences = {}
    # for machine_name, group in data:
    #     feature_columns = [col for col in group.columns if col not in non_feature_columns]
    #     features = group[feature_columns].values
    #     labels = group['label'].values
    #     sequences[machine_name] = (features, labels)
    #
    # # 获取最大序列长度
    # max_length = max(len(seq[0]) for seq in sequences.values())
    #
    # # 填充序列
    # padded_features = {}
    # padded_labels = {}
    #
    # for machine_name, (features, labels) in sequences.items():
    #     padded_features[machine_name] = pad_sequences(features, maxlen=max_length, padding='post', dtype='float32')
    #     padded_labels[machine_name] = pad_sequences([labels], maxlen=max_length, padding='post', dtype='int32')[0]
    # print(features.shape)
    # print(features)
    #
    # X = np.array([padded_features[machine] for machine in padded_features])
    # y = np.array([padded_labels[machine] for machine in padded_labels])
    #
    # print("输入形状：", X.shape)  # (num_machines, max_length, num_features)
    # print("标签形状：", y.shape)  # (num_machines, max_length)

    # 更新编码器并保存
    # for field, encoder in encoders.items():
    #     print(f"字段名: {field}")
    #     print(f"编码器内容: {encoder.classes_}\n")
    with open(encoder_path, 'wb') as f:
        pickle.dump(encoders, f)

    print(f"\n格式化后表 shape {data.shape}")
    print(f"格式化后表格式：共{len(data)}行\t共{len(data.columns)}列")
    for col in data.columns:
        print(f"格式化后列名：{data[col].name}\t列类型：{data[col].dtype}")

    # 4. 划分数据集
    X = data.drop(['label'], axis=1)   # axis=0：沿着行操作; axis=0：沿着列操作
    # TODO 多标签训练
    y = data['label']
    print(f"X shape：{X.shape}")
    print(f"y shape：{y.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 5. 保存到本地
    # TODO 待分别生成train、test数据，切割存量与增量
    # 使用字典格式保存预处理后数据
    torch.save({'X': X_train, 'y': y_train}, project_dir + 'dataset/SERVER/train_dataset.pt')
    torch.save({'X': X_test, 'y': y_test}, project_dir + 'dataset/SERVER/test_dataset.pt')
    X_train.to_csv(project_dir + "dataset/SERVER/train_features.csv", index=False, encoding="utf-8-sig")
    y_train.to_csv(project_dir + "dataset/SERVER/train_labels.csv", index=False, encoding="utf-8-sig")
    X_test.to_csv(project_dir + "dataset/SERVER/test_features.csv", index=False, encoding="utf-8-sig")
    y_test.to_csv(project_dir + "dataset/SERVER/test_labels.csv", index=False, encoding="utf-8-sig")

    print(f"训练[样本数, 特征数]{X.shape}")  # (样本数, 高度, 宽度)
    print(f"训练[样本数, 标签数]{y.shape}")  # (样本数, 高度, 宽度)

    # 5. 转换成图片用于ResNet18网络训练
    # 生成图像数据集，不支持字符
    # img_size = 32
    # img_channel = 2
    # image_data = np.array([generate_image(data.iloc[i], img_size, img_channel) for i in range(len(data))])
    # image_labels = data['label'].to_numpy()
    #
    # # 划分数据集
    # train_images, test_images, train_labels, test_labels = train_test_split(image_data, image_labels, test_size=0.2,
    #                                                                         random_state=42)
    #
    # # 保存训练集和测试集
    # save_images_to_directory(train_images, train_labels, "./pytorchDemo/dataset/SERVER", "train")
    # save_images_to_directory(test_images, test_labels, "./pytorchDemo/dataset/SERVER", "test")


# 初始化label编码器
def init_encoders(encoder_path, columns):
    # 加载编码器或者创建
    if os.path.exists(encoder_path):
        # 如果文件存在，加载编码器字典
        with open(encoder_path, 'rb') as f:
            encoders = pickle.load(f)
    else:
        # 如果文件不存在，创建一个新的空字典并保存
        encoders = {}
        with open(encoder_path, 'wb') as f:
            pickle.dump(encoders, f)

    # 初始化编码器新字段
    for col in columns:
        if col not in encoders:
            encoder = LabelEncoder()
            encoder.classes_ = []  # 初始化为空类别
            encoders[col] = encoder

    return encoders


def generate_image(row, img_size, image_channel):
    """
    将一行特征转化为图像
    :param image_channel:
    :param row: 数据行
    :param img_size: 图像尺寸 (img_size x img_size)
    :return: 多通道图像 (C, H, W)
    """
    if image_channel == 1:
        feature_vector = row.to_numpy(dtype=np.float32)  # 转化为 numpy 数组
        # 填充或截断以匹配图像尺寸
        padded_vector = np.pad(
            feature_vector,
            (0, img_size * img_size - len(feature_vector)),
            mode='constant')
        truncated_vector = padded_vector[:img_size * img_size]  # 截断为 img_size^2
        # 重塑为图像形状 (H, W)
        image = truncated_vector.reshape((img_size, img_size))
        # 增加通道维度为 (1, H, W)
        image = np.expand_dims(image, axis=0)

        return image
    else:
        # 提取特征组：CPU 通道, 内存 通道
        cpu_features = row[
            ["机器名", "业务IP", "归属系统", "归属微服务", "系统级别", "进程", "规格", "机房", "网络区域", "芯片架构",
             "最高CPU利用率", "CPU利用率(>20%)时间占比", "CPU利用率(>30%)时间占比", "year", "month",
             "day"]].to_numpy()  # CPU特征
        memory_features = row[
            ["机器名", "业务IP", "归属系统", "归属微服务", "系统级别", "进程", "规格", "机房", "网络区域", "芯片架构",
             "最高内存利用率", "year", "month", "day"]].to_numpy()  # 内存特征

        # 合并所有特征并填充到 img_size^2
        all_features = np.concatenate([cpu_features, memory_features])
        padded_features = np.pad(all_features, (0, img_size * img_size - len(all_features)), mode='constant')
        truncated_features = padded_features[:img_size * img_size]  # 确保特征数量不超过图像尺寸

        # 将特征映射到多通道
        cpu_channel = truncated_features[:img_size * img_size].reshape((img_size, img_size))
        memory_channel = truncated_features[:img_size * img_size].reshape((img_size, img_size))

        # 合并通道
        image = np.stack([cpu_channel, memory_channel])
        return image


def save_images_to_directory(images, labels, save_dir, dataset_type):
    """
    保存图像到本地目录，按标签分类存储
    :param images: 图像数据 (N, 2, H, W)
    :param labels: 标签列表 (N,)
    :param save_dir: 根目录
    :param dataset_type: 'train' or 'test'
    """
    for idx, (image, label) in enumerate(zip(images, labels)):
        # 创建对应标签的文件夹
        label_dir = os.path.join(save_dir, dataset_type, f"label_{label}")
        os.makedirs(label_dir, exist_ok=True)

        # 保存图像
        image_path = os.path.join(label_dir, f"{idx}.png")
        # 转换为PIL Image（去掉单通道维度）
        image = Image.fromarray((image[0] * 255).astype(np.uint8))
        image.save(image_path)


if __name__ == '__main__':
    make_server_data()
