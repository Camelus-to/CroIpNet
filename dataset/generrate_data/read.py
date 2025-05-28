import numpy as np
from sklearn.preprocessing import StandardScaler

def check_nan():
    data = np.load('processed_data1.npy', allow_pickle=True).item()
    # data = np.load('processed_data.npy', allow_pickle=True).item()

    final_timeseires = data["timeseires"]
    final_pearson = data["corr"]
    labels = data["label"]
    site = data['site']

    # 检查是否有 NaN 值
    has_nan_in_timeseires = np.isnan(final_timeseires).any()
    has_nan_in_pearson = np.isnan(final_pearson).any()
    has_nan_in_labels = np.isnan(labels).any()
    # has_nan_in_site = np.isnan(site).any()

    # 输出检查结果
    print(f"final_timeseires has NaN: {has_nan_in_timeseires}")
    print(f"final_pearson has NaN: {has_nan_in_pearson}")
    print(f"labels has NaN: {has_nan_in_labels}")
    # print(f"site has NaN: {has_nan_in_site}")
    cc = 0
from nilearn import connectome
def check_abide():
    data = np.load(r'E:\Technolgy_learning\Learning_code\AD\BrainNetworkTransformer-View-Bert\source\abide.npy', allow_pickle=True).item()
    final_timeseires = data["timeseires"]
    final_pearson = data["corr"]
    labels = data["label"]
    site = data['site']
    first = final_timeseires[0]
    # 对信号值进行归一化
    scaler = StandardScaler()
    first_strad = scaler.fit_transform(first)

    fc = final_pearson[0]
    conn_measure = connectome.ConnectivityMeasure(kind='correlation')
    connectivity = conn_measure.fit_transform([first.T])
    connectivity_strad = conn_measure.fit_transform([first_strad.T])
    fc_matrix = np.corrcoef(first)
    connectivity = np.squeeze(connectivity)
    connectivity[np.diag_indices_from(connectivity)] = 0
    connectivity_s = scaler.fit_transform(connectivity)
    c = 0

check_abide()