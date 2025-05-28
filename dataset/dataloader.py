import torch
import torch.utils.data as utils
from omegaconf import DictConfig, open_dict
from typing import List
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import torch.nn.functional as F
import pickle
import pandas as pd
import os
from torch_geometric.data import Data
# from sklearn.preprocessing import StandardScaler
from torch_geometric.data import DataLoader


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def init_stratified_dataloader(dataset_config):
    dataset = dataset_config['data']['dataset']  # 获取数据集名称

    # 根据数据集选择对应的 time_series 路径
    time_series = dataset_config['data']['time_series_paths'].get(dataset, None)
    if time_series is None:
        raise ValueError(f"Dataset {dataset} not found in time_series_paths.")
    print(f"Using dataset: {dataset}")
    print(f"Using time series data from: {time_series}")
    data = np.load(time_series, allow_pickle=True).item()

    # final_fc = data["signals"]  # 这个是时间序列的信号值
    final_pearson = data["corr"]

    signals = data["signals"]
    if dataset == 'ABIDEII':
        signals = signals.transpose(0, 2, 1)
    labels = data["label"]
    ages = data["age"]
    genders = data["sex"]
    site = data["site"]
    # 将站点转换为 DataFrame 并且对站点列进行 one-hot 编码
    # site_df = pd.DataFrame({"site": site})
    # site_one_hot = pd.get_dummies(site_df, columns=["site"], prefix="site").astype(int)
    # site_one_hot = site.to_numpy()
    labels = torch.tensor(labels)
    labels = F.one_hot(labels.to(torch.int64))
    labels = labels.numpy()
    _, _, timeseries = final_pearson.shape  # 第一个维度是批次大小，第二个维度是脑区数, 最后一个是时间点

    _, node_size, node_feature_size = final_pearson.shape

    # scaler = StandardScaler(mean=np.mean(final_fc), std=np.std(final_fc))
    #
    # final_fc = scaler.transform(final_fc)


    final_pearson = np.array(final_pearson, dtype=np.float32)  # 确保是 float32
    signals_yuan = np.array(signals, dtype=np.float32)  # 确保是 float32
    # 处理信号值
    # ** 仅当数据集是 ADNI 时，进行类别筛选 **
    if dataset == 'ADNI':

        selected_indices = np.where(labels[:, 1] == 0)

        selected_indices = np.where(labels[:, 1] == 0)[0]

        final_pearson = final_pearson[selected_indices]
        signals = signals[selected_indices]
        # labels = labels[selected_indices, :2]  # 仅保留前两个类别
        labels = labels[selected_indices][:, [0, 2]]  # 保留第一列和第三列
        ages = ages[selected_indices]
        genders = genders[selected_indices]
        site = site[selected_indices]
    pseudo = []
    for i in range(len(final_pearson)):
        pseudo.append(np.diag(np.ones(final_pearson.shape[1])))

    if 'cc200' in dataset_config['data']['atlas']:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 200, 200))
    elif 'aal' in dataset_config['data']['atlas']:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 116, 116))
    elif 'cc400' in dataset_config['data']['atlas']:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 392, 392))
    elif 'Harvard' in dataset_config['data']['atlas']:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 96, 96))
    else:
        pseudo_arr = np.concatenate(pseudo, axis=0).reshape((-1, 111, 111))

    signals = np.array(signals, dtype=np.float32)
    site_one_hot = np.array(site, dtype=np.float32)
    final_pearson, signals, labels, pseudo_arr, ages, genders, site_one_hot = [
        torch.from_numpy(d).float() for d in (
            final_pearson, signals, labels, pseudo_arr, ages, genders, site_one_hot)]  # 转为torch向量

    dataset = utils.TensorDataset(
        final_pearson,
        signals,
        labels,
        pseudo_arr,
        ages,
        genders,
        site_one_hot
    )

    length = len(dataset)
    train_length = int(length * dataset_config.dataset.train_set * dataset_config.datasz.percentage)
    val_length = int(length * dataset_config.dataset.val_set)

    if dataset_config.datasz.percentage == 1.0:
        test_length = length - train_length - val_length
    else:
        test_length = int(length * (1 - dataset_config.dataset.val_set - dataset_config.dataset.train_set))

    with open_dict(dataset_config):
        dataset_config.steps_per_epoch = (train_length - 1) // dataset_config['data']['batch_size'] + 1
        dataset_config.total_steps = dataset_config.steps_per_epoch * dataset_config.training.epochs

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_length, length - train_length])

    train_dataloader = utils.DataLoader(
        train_dataset, batch_size=dataset_config['data']["batch_size"], shuffle=True, drop_last=True)

    val_dataloader = utils.DataLoader(
        val_dataset, batch_size=dataset_config['data']["batch_size"], shuffle=True, drop_last=True)

    test_dataloader = val_dataloader
    return (train_dataloader, val_dataloader, test_dataloader), node_size, node_feature_size, timeseries, (
    train_dataset, val_dataset)
