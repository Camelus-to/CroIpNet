import numpy as np
import torch
from .preprocess import StandardScaler
from omegaconf import DictConfig, open_dict


def load_abide_data(cfg: DictConfig):

    data = np.load(cfg.dataset.path, allow_pickle=True).item()
    size = data['corr'].shape[0]
    final_timeseires = np.random.rand(size, 116, 116).astype(np.float32)
    final_pearson = data["corr"]
    dynamic_final_pearson = data["dcorr"]
    labels = data["label"]
    site = data['site']

    # final_timeseires = np.array(final_timeseires, dtype=np.float32)  # 确保是 float32
    final_pearson = np.array(final_pearson, dtype=np.float32)  # 确保是 float32
    dynamic_final_pearson = np.array(dynamic_final_pearson, dtype=np.float32)  # 确保是 float32
    labels = np.array(labels, dtype=np.int64)  # 确保标签是 int64
    # 先不进行归一化
    # scaler = StandardScaler(mean=np.mean(
    #     final_timeseires), std=np.std(final_timeseires))
    #
    # final_timeseires = scaler.transform(final_timeseires)

    final_timeseires, final_pearson, dynamic_pearson, labels = [torch.from_numpy(
        data).float() for data in (final_timeseires, final_pearson, dynamic_final_pearson, labels)]

    with open_dict(cfg):

        cfg.dataset.node_sz, cfg.dataset.node_feature_sz = final_pearson.shape[1:]
        cfg.dataset.timeseries_sz = final_timeseires.shape[2]

    return final_timeseires, final_pearson, dynamic_pearson, labels, site


