import os
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat
import scipy


# 处理动态功能连接矩阵文件
def process_dynamic_connectivity_files(input_dir, dynamic_matrix_dir, output_dir, df):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    signals_path_yuan = r"E:\Technolgy_learning\Learning_code\New_Stage\shujuji\ADNI\Signals"
    # 初始化列表存储最终输出的内容
    FC_list = []
    FC_dynamic_list = []
    Label_list = []
    Site_list = []
    Filename_list = []
    sex_list = []
    Signals_list = []
    age_list = []
    FIQ_list = []
    VIQ_list = []
    PIQ_list = []
    # 遍历动态功能连接矩阵文件夹中的每个样本文件
    site_onehot = pd.get_dummies(df['Site']).astype(int)  # 创建 one-hot 编码
    for filename in os.listdir(dynamic_matrix_dir):
        if filename.endswith('.npy'):
            # 构建文件路径并加载数据
            static_file_path = os.path.join(input_dir, filename)
            file_path = os.path.join(dynamic_matrix_dir, filename)
            static_matrix_dir = np.load(static_file_path, allow_pickle=True)
            np.fill_diagonal(static_matrix_dir, 1)
            static_matrix_dir = static_matrix_dir
            signals_name = filename[:-4] + '.mat'
            signals_path = os.path.join(signals_path_yuan, signals_name)
            ROISignals_data = scipy.io.loadmat(signals_path)
            signals = ROISignals_data['ROISignals'].T
            # 样本名（去掉文件后缀）
            sample_id = filename[:-4]
            df['Subject'] = df['Subject'].astype(str).str.strip()  # 确保SUB_ID是字符串类型并去除空格
            # 查找样本的分类标签和站点信息
            match = df[df['idid'] == str(sample_id)]
            if len(match) == 0:
                print(f"No match found for {sample_id}")
                continue

            # 获取分类标签 上面的是随机生成标签在0和1之间 下面的是从表中获取
            # y = np.random.randint(0, 2)  # 0 或 1
            y = match.iloc[0]['Group_x']
            site_id = match.iloc[0]['Site']  # 获取站点ID

            # 将站点信息转换为 one-hot 编码
            site_onehot_encoding = site_onehot.loc[match.iloc[0].name].values.flatten()  # 只取匹配的第一行
            sex = match.iloc[0]['Sex_x']
            age = match.iloc[0]['Age_x']


            # 将动态功能连接矩阵、标签和文件名添加到列表中
            FC_list.append(static_matrix_dir)
            # FC_dynamic_list.append(dynamic_matrices)
            Label_list.append(y)
            Site_list.append(site_onehot_encoding)
            Filename_list.append(sample_id)
            sex_list.append(sex)
            age_list.append(age)

            Signals_list.append(signals)

            print(f"Processed {sample_id}, loaded dynamic matrices.")

    # 转换列表为数组
    FC_list = np.array(FC_list, dtype=object)
    FC_dynamic = np.array(FC_dynamic_list, dtype=object)
    Label_array = np.array(Label_list, dtype=np.int64)
    Site_array = np.array(Site_list, dtype=object)
    Signals_list = np.array(Signals_list, dtype=object)
    Sex_array = np.array(sex_list, dtype=np.int64)
    Age_array = np.array(age_list, dtype=np.int64)
    Filename_array = np.array(Filename_list, dtype=object)

    # 将动态功能连接矩阵、标签、站点信息和文件名保存到字典中
    data_dict = {
        'corr': FC_list,  # 静态功能连接矩阵
        'label': Label_array,  # 标签
        'site': Site_array,  # 站点
        'sub': Filename_array,  # 文件名
        'age': Age_array,
        'sex': Sex_array,
        'signals': Signals_list
    }

    # 保存字典为 .npy 文件
    output_file = os.path.join(output_dir, 'connectivity_data.npy')
    np.save(output_file, data_dict)

    print(f"All data saved to {output_file}")


# 主函数
if __name__ == "__main__":
    # 动态功能连接矩阵文件夹路径
    static_matrix_dir = r'E:\Technolgy_learning\Learning_code\New_Stage\shujuji\ADNI\FC'
    dynamic_matrix_dir = r'E:\Technolgy_learning\Learning_code\New_Stage\shujuji\ADNI\FC'
    output_dir = (r'E:\Technolgy_learning\Learning_code\New_Stage\KMGCN-main\KMGCN-yanxu\dataset\generrate_abide'
                  r'\processed')  # 最终数据输出文件夹

    # 读取 CSV 文件
    df = pd.read_csv(r'E:\Technolgy_learning\Learning_code\New_Stage\shujuji\ADNI\merger.csv')

    df['Site'] = df['Subject'].apply(lambda x: x.split('_')[0])
    df['idid'] = df['Subject'].astype(str) + '_' + df['Visit'].astype(str)
    df['Group_x'] = df['Group_x'].replace({'CN': '0', 'MCI': '1', 'LMCI': '1', 'EMCI': '1', 'AD': '2'})
    df['Sex_x'] = df['Sex_x'].replace({'F': '0', 'M': '1'})
    process_dynamic_connectivity_files(static_matrix_dir, dynamic_matrix_dir, output_dir, df)
