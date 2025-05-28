import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel  # 可以使用RBF核计算相似度矩阵
from torch.nn import Parameter
import torch.nn as nn
from typing import Tuple


class SpectralClusteringWithNodeFeatures(nn.Module):
    def __init__(self, cluster_number: int, gamma: float = 1.0):
        """
        Spectral Clustering module, which performs spectral clustering on a similarity matrix
        and returns both node features and cluster assignments.

        :param cluster_number: number of clusters
        :param gamma: RBF kernel parameter for computing similarity matrix
        """
        super(SpectralClusteringWithNodeFeatures, self).__init__()
        self.cluster_number = cluster_number
        self.gamma = gamma

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform spectral clustering on the input batch of data, and return node features
        and cluster assignments.

        :param batch: [batch_size, num_nodes, feature_dim] similarity matrix or feature matrix
        :return: Tuple of [batch_size, num_nodes, feature_dim] node features and
                 [batch_size, num_nodes] cluster assignments
        """
        batch_size, num_nodes, feature_dim = batch.shape

        all_assignments = []
        all_node_features = []

        for i in range(batch_size):
            # Step 1: 计算相似度矩阵，可以用 RBF 核或者其他方式计算
            similarity_matrix = rbf_kernel(batch[i].detach().cpu().numpy(), gamma=self.gamma)

            # Step 2: 计算拉普拉斯矩阵
            degree_matrix = torch.diag(torch.sum(torch.tensor(similarity_matrix), dim=1))
            laplacian_matrix = degree_matrix - torch.tensor(similarity_matrix)

            # Step 3: 特征值分解，获取前 k 个最小特征值对应的特征向量
            eigvals, eigvecs = torch.linalg.eigh(laplacian_matrix)
            eigvecs = eigvecs[:, :self.cluster_number]  # 取出前 cluster_number 个特征向量作为节点的新特征

            # 将特征向量作为新的节点特征
            node_features = torch.tensor(eigvecs).to(batch.device)
            all_node_features.append(node_features)

            # Step 4: 在低维空间进行 K-means 聚类
            kmeans = KMeans(n_clusters=self.cluster_number)
            kmeans.fit(node_features.detach().cpu().numpy())
            assignments = torch.tensor(kmeans.labels_).to(batch.device)
            all_assignments.append(assignments)

        # 将所有 batch 的节点特征和聚类结果拼接
        all_node_features = torch.stack(all_node_features, dim=0)  # [batch_size, num_nodes, feature_dim]
        all_assignments = torch.stack(all_assignments, dim=0)  # [batch_size, num_nodes]

        return all_node_features, all_assignments

    def loss(self, assignments, target_distribution):
        """
        Loss function can be modified for spectral clustering, but in basic form,
        spectral clustering does not have a direct loss function like DEC.
        You can use target distribution to modify the assignments.
        """
        return F.mse_loss(assignments, target_distribution)  # 可以使用 MSE loss 或者根据需求调整损失函数


# 生成随机的 batch 数据，假设 batch_size=2, num_nodes=116, feature_dim=116
batch_size = 2
num_nodes = 116
feature_dim = 116

# 随机生成的相似度矩阵（可以认为是 Transformer 处理后的相似度）
random_data = torch.rand(batch_size, num_nodes, feature_dim)

# 创建谱聚类模型的实例，假设要分成5个簇
model = SpectralClusteringWithNodeFeatures(cluster_number=5, gamma=1.0)

# 将数据传入模型进行聚类
node_features, assignments = model(random_data)

# 输出节点特征和聚类结果
print("Node Features Shape:", node_features.shape)  # 预期输出：[batch_size, num_nodes, cluster_number]
print("Cluster Assignments Shape:", assignments.shape)  # 预期输出：[batch_size, num_nodes]

# 输出第一个batch的聚类结果
print("Cluster Assignments for first batch:", assignments[0])
