import pandas as pd
import torch

# 1. 读取训练数据 与 测试数据
def read_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    # data构成了train和test的全部的数据
    data = pd.concat((train_df.iloc[:, 1:-1], test_df.iloc[:, 1:]))
    print("data.shape=", data.shape)
    # 删除缺失值超过1/3的特征, 仅以train_df
    missing = data.isnull().sum()
    missing = missing[missing > data.shape[0] // 3]
    mov_f = list(dict(missing).keys())
    data = data.drop(labels=mov_f, axis=1)
    print("after delete nan features, data.shape=", data.shape)
    # 对numeric对象进行均值归一化
    numeric = [num for num in data.columns if data.dtypes[num] != 'object']
    data[numeric] = data[numeric].apply(lambda x: (x-x.mean())/x.std())
    data[numeric] = data[numeric].fillna(0)
    # 对category特征进行onehot编码, 这将大大增加特征数量
    data = pd.get_dummies(data, dummy_na=True)
    print("after one-hot operation, data.shape=", data.shape)
    # 转为tensor后返回
    train_size = train_df.shape[0]
    train_features = torch.tensor(data[:train_size].values, dtype=torch.float32)
    test_features = torch.tensor(data[train_size:].values, dtype=torch.float32)
    train_labels = torch.tensor(train_df["SalePrice"].values.reshape(-1, 1), dtype=torch.float32)
    return train_features, train_labels, test_features
