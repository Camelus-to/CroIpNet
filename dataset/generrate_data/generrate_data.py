import os
import numpy as np
from scipy.io import loadmat, savemat
import pandas as pd

df = pd.read_csv(
    r'E:\Technolgy_learning\Learning_code\AD\AD_Bert\data\Phenotypic_V1_0b_preprocessed1.csv')
# 输入文件夹路径
input_dir = "E:\Technolgy_learning\Learning_code\AD\AD_Bert\data\ROISignals"
# 输出文件夹路径
output_dir = "E:\Technolgy_learning\Learning_code\AD\AD_Bert\data\FC"
FC_list = []
ROI_list = []
Label_list = []
Site_list = []
# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历输入目录下的所有 .mat 文件
for filename in os.listdir(input_dir):
    if filename.endswith(".mat"):
        # 构建文件的完整路径
        file_path = os.path.join(input_dir, filename)

        # 加载 .mat 文件
        ROISignals_data = loadmat(file_path)
        match = df[df['subject'] == str(filename)[:5]]
        y = match.iloc[0]['DX_GROUP']
        site = match.iloc[0]['SITE_ID']
        # 假设时间序列数据在变量 'time_series' 中
        # 根据实际情况替换为实际的变量名
        if 'ROISignals' in ROISignals_data:
            time_series = ROISignals_data['ROISignals']
            time_series = time_series.T
            # 计算相关性矩阵
            feature_path = os.path.join(output_dir, filename)
            data = loadmat(feature_path)
            fc_matrix = data['fc_matrix']
            has_nan_in_timeseires = np.isnan(fc_matrix).any()
            if has_nan_in_timeseires:
                fc_matrix = np.nan_to_num(fc_matrix, nan=0.0)
                print(filename)
            fc_matrix[np.diag_indices_from(fc_matrix)] = 0
            FC_list.append(fc_matrix)
            ROI_list.append(time_series)
            Label_list.append(y)
            Site_list.append(site)
            # 构建输出文件路径
            output_file_path = os.path.join(output_dir, filename)

            # 保存相关性矩阵到新的 .mat 文件
            # savemat(output_file_path, {'fc_matrix': fc_matrix})

            print(f"Processed {filename}, saved FC matrix to {output_file_path}")
        else:
            print(f"Variable 'time_series' not found in {filename}")

# FC_array = np.array(FC_list)   # 转换为99x8x9的数组
# 对ROI_list进行处理
min_dim = min([roi.shape[1] for roi in ROI_list])
# 第二步：根据最小维度裁剪数组
processed_ROI_list = []
for roi in ROI_list:
    if roi.shape[1] > min_dim:
        # 裁剪第二个维度到前min_dim列
        processed_roi = roi[:, :min_dim]
    else:
        processed_roi = roi  # 如果第二个维度 <= min_dim，保持不变
    processed_ROI_list.append(processed_roi)
# 将处理后的列表转换为object数组
processed_ROI_array = np.array(processed_ROI_list, dtype=object)
FC_array = np.array(FC_list, dtype=object)
Label_array = np.array(Label_list, dtype=object)
Site_array = np.array(Site_list, dtype=object)

processed_ROI_array = np.array(processed_ROI_array, dtype=np.float32)
FC_array = np.array(FC_array, dtype=np.float32)
Label_array = np.array(Label_array, dtype=np.int64)

# 打包为一个字典并保存
data_dict = {'corr': FC_array, 'timeseires': processed_ROI_array, 'label': Label_array, 'site': Site_array}
np.save('processed_data1.npy', data_dict)

print("All files processed.")
