import torch
import numpy as np
data = np.load(r"E:\Technolgy_learning\Learning_code\New_Stage\KMGCN-main\KMGCN-yanxu\dataset\generrate_abide"
               r"\processed\connectivity_data1.npy",allow_pickle=True).item()
data2 = np.load(r"E:\Technolgy_learning\Learning_code\New_Stage\KMGCN-main\KMGCN-yanxu\dataset\generrate_abide\processed\connectivity_data_abide_cc200.npy",allow_pickle=True).item()
data2['signals'] = data2['signals'].transpose(0, 2, 1)
if torch.cuda.is_available():
    print("CUDA is available. GPU will be used.")
else:
    print("CUDA is not available. Using CPU.")
