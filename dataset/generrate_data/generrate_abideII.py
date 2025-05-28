import os
import numpy as np
import pandas as pd


# FIQ: Full-scale IQ, the total IQ score calculated through psychological testing.
#
# VIQ: Verbal IQ, the IQ score measuring the subject's language ability.
#
# PIQ: Performance IQ, the IQ score measuring the subject's non-verbal ability.
# 处理动态功能连接矩阵文件
def process_dynamic_connectivity_files(input_dir, dynamic_matrix_dir, output_dir, df):
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    signals_path_yuan = r"E:\Technolgy_learning\Learning_code\New_Stage\shujuji\ABIDE_II\Signals"
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
    site_onehot = pd.get_dummies(df['Site']).astype(int)
    for filename in os.listdir(signals_path_yuan):
        if filename.endswith('.npy'):
            # 构建文件路径并加载数据
            static_file_path = os.path.join(input_dir, filename)
            file_path = os.path.join(dynamic_matrix_dir, filename)
            dynamic_matrices = np.load(file_path, allow_pickle=True)  # 加载动态功能连接矩阵
            dynamic_matrices = dynamic_matrices
            static_matrix_dir = np.load(static_file_path, allow_pickle=True)
            static_matrix_dir = static_matrix_dir
            signals_path = os.path.join(signals_path_yuan, filename)
            signals = np.load(signals_path, allow_pickle=True)[:110, :]
            # 样本名（去掉文件后缀）
            sample_id = filename[:-4]
            df['Subject'] = df['Subject'].astype(str).str.strip()  # 确保SUB_ID是字符串类型并去除空格
            # 查找样本的分类标签和站点信息
            match = df[df['Subject'] == str(sample_id)]
            if len(match) == 0:
                print(f"No match found for {sample_id}")
                continue

            # 获取分类标签 上面的是随机生成标签在0和1之间 下面的是从表中获取
            y = match.iloc[0]['DX_GROUP']
            site_onehot_encoding = site_onehot.loc[match.index].values.flatten()
            sex = match.iloc[0]['SEX']
            age = match.iloc[0]['AGE_AT_SCAN']

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
    # FIQ_array = np.array(FIQ_list, dtype=np.int64)
    # VIQ_array = np.array(VIQ_list, dtype=np.int64)
    # PIQ_array = np.array(PIQ_list, dtype=np.int64)
    Filename_array = np.array(Filename_list, dtype=object)

    # 将动态功能连接矩阵、标签、站点信息和文件名保存到字典中
    data_dict = {
        'corr': FC_list,  # 静态功能连接矩阵
        # 'dcorr': FC_dynamic,  # 动态功能连接矩阵
        'label': Label_array,  # 标签
        'site': Site_array,  # 站点
        'sub': Filename_array,  # 文件名
        'age': Age_array,
        'sex': Sex_array,
        # 'fiq': FIQ_array,
        # 'viq': VIQ_array,
        # 'piq': PIQ_array,
        'signals': Signals_list
    }

    # 保存字典为 .npy 文件
    output_file = os.path.join(output_dir, 'connectivity_data_all.npy')
    np.save(output_file, data_dict)

    print(f"All data saved to {output_file}")


# 主函数
if __name__ == "__main__":
    # 动态功能连接矩阵文件夹路径
    static_matrix_dir = r'E:\Technolgy_learning\Learning_code\New_Stage\shujuji\ABIDE_II\FC_static'
    dynamic_matrix_dir = r'E:\Technolgy_learning\Learning_code\New_Stage\shujuji\ABIDE_II\FC_static'
    output_dir = r'E:\Technolgy_learning\Learning_code\New_Stage\KMGCN-main\KMGCN-yanxu\dataset\generrate_abide\processed'  # 最终数据输出文件夹

    # 读取 CSV 文件
    df = pd.read_csv(
        r'E:\Technolgy_learning\Learning_code\New_Stage\shujuji\ABIDE_II\ABIDE2.csv',
    )
    # df['Label'] = df['ID'].apply(lambda x: x.split('-')[1])  # 提取标签
    process_dynamic_connectivity_files(static_matrix_dir, dynamic_matrix_dir, output_dir, df)
