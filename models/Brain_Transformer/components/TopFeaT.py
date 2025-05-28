import torch
import torch.nn as nn
import gudhi as gd
import numpy as np
import matplotlib.pyplot as plt
import gudhi.representations
from sklearn_tda import *
import scipy.sparse as sp
from numpy import polynomial as P


# 定义 MyPHlayer 模型
class MyPHlayer(nn.Module):
    def __init__(self, resolution, land, dim_obj, nROI):
        super(MyPHlayer, self).__init__()
        self.resolution = resolution
        self.nROI = nROI
        self.nland = land
        self.sampling_control = 50
        self.dim_obj = dim_obj  # 0-dimensional and 1-dimensional persistent feature

    def forward(self, x, edge_index, edge_attr):

        # 受输入特征 x、边索引 edge_index 和边属性 edge_attr，然后调用 Compute_Persistent_Feats 方法计算持久性特征，并将结果转换为PyTorch张量返回。
        land, betti, curve = self.Compute_Persistent_Feats(x, edge_index, edge_attr, self.resolution, self.nland,
                                                           self.dim_obj)
        land_PH1_feats = torch.from_numpy(np.squeeze(np.array(land, dtype=np.float32)))
        betti_feats = torch.from_numpy(np.squeeze(np.array(betti, dtype=np.float32)))
        curve_PH0_feats = torch.from_numpy(np.squeeze(np.array(curve, dtype=np.float32)))

        return land_PH1_feats, betti_feats, curve_PH0_feats

    # 这个方法用于从大矩阵 A 中提取大小为 M 的块对角线。主要用于处理脑网络数据中的块对角线矩阵。提取对角线元素
    def fix_negative_values(self, matrix):
        # 将所有负值设置为零
        matrix[matrix < 0] = 0
        return matrix

    def check_distance_matrix(self, matrix):
        if np.any(np.isnan(matrix)):
            print("Distance matrix contains NaN values.")
        if np.any(np.isinf(matrix)):
            print("Distance matrix contains Inf values.")
        if not np.allclose(matrix, matrix.T):
            print("Distance matrix is not symmetric.")
        if np.any(matrix < 0):
            print("Distance matrix contains negative values.")
        else:
            print("Distance matrix is valid.")

    def extract_block_diag(self, A, M, k=0):
        """Extracts blocks of size M=numofROIs from the kth diagonal of brain connectivity A,
        whose size must be a multiple of M. 从一个矩阵 A 中提取指定对角线的大小为 的M 块，A 被假设为一个大脑连接矩阵"""
        # Check that the matrix can be block divided； k = 0 表示主对角线，k > 0 表示上对角线，k < 0 表示下对角线 在这里K是等于0的
        if A.shape[0] != A.shape[1] or A.shape[0] % M != 0:
            print('Matrix must be square and a multiple of block size')
        # Assign indices for offset from main diagonal
        if abs(k) > M - 1:
            print('kth diagonal does not exist in matrix')
        elif k > 0:
            ro = 0
            co = abs(k) * M
        elif k < 0:
            ro = abs(k) * M
            co = 0
        else:
            ro = 0
            co = 0

        blocks = np.array([A[i + ro:i + ro + M, i + co:i + co + M]
                           for i in range(0, len(A) - abs(k) * M, M)])
        return blocks

    # nland：持久性同调地貌（Landscapes）的数量。dim_obj：持久性同调维度对象（0维和1维）
    # 输出特征Land_feature：持久性同调地貌特征。betti_num：Betti数曲线特征。curve_feature：曲线特征。
    def Compute_Persistent_Feats(self, x, edge_index, edge_attr, res, nland, dim_obj):
        adj_gro = sp.coo_matrix(
            (edge_attr.squeeze().numpy(), (edge_index.numpy()[0, :], edge_index.numpy()[1, :])),
            shape=(x.shape[1], x.shape[1]), dtype="float32").toarray()
        # 提取块对角线；提取邻接矩阵的块对角线子矩阵。adj_gro是一个大的矩阵，是812*812的好像。然后这个方法是将更大的矩阵分割成更小的块,然后提取各个矩阵的对角线上的这些块
        # adj_gro本身也就是一个只有对角线上才有元素，除了对角线以外
        Land_feature = []
        betti_num = []
        curve_feature = []
        for batch in range(x.shape[0]):  # 按批次遍历
            adj_sub = self.extract_block_diag(adj_gro, self.nROI)
            for i in range(adj_sub.shape[0]):
                correlation_matrix = np.array(adj_sub[i], float)
                distance_matrix = 1 - correlation_matrix
                distance_matrix = self.fix_negative_values(distance_matrix)  # 修正负值
                self.check_distance_matrix(distance_matrix)  # 检查距离矩阵

                rips_complex = gd.RipsComplex(distance_matrix=distance_matrix, max_edge_length=10)
                simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)

                # 计算持久性同调
                Diags = simplex_tree.persistence(homology_coeff_field=11, min_persistence=0)
                PH_0 = simplex_tree.persistence_intervals_in_dimension(0)
                PH_1 = simplex_tree.persistence_intervals_in_dimension(1)

                # 计算Betti-0曲线：
                if dim_obj == 0:
                    diags_dim0 = DiagramSelector(use=True, point_type="finite"). \
                        fit_transform([PH_0])
                    BC = BettiCurve(resolution=res)
                    Betti_num_curve = BC.fit_transform(diags_dim0)
                    X_axis = np.arange(res).reshape(-1, 1).squeeze()
                    p = P.polynomial.Polynomial.fit(X_axis, Betti_num_curve.squeeze(), deg=500)
                    yvals = p(X_axis)  # fitting curve
                    fderiv = abs(p.deriv()(X_axis))[
                             self.sampling_control:(res - self.sampling_control)]  # derivative curve
                    betti_num.append(yvals)
                    curve_feature.append(fderiv)

                # 计算持久性同调地貌（Landscapes）
                diags_dim1 = DiagramSelector(use=True, point_type="finite"). \
                    fit_transform([PH_1])
                LS = gd.representations.Landscape(num_landscapes=nland, resolution=res)
                L = LS.fit_transform(diags_dim1)
                land_average = np.average(L.reshape(nland, res), 0)
                Land_feature.append(land_average)

        return Land_feature, betti_num, curve_feature
