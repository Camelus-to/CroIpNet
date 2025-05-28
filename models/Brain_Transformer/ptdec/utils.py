import torch
from torch_geometric.data import Data


def create_graph_data(node_feature: torch.Tensor, assignments: torch.Tensor):
    """
    生成图数据，节点特征由原始特征和聚类分配结果拼接，边根据聚类分配与注意力机制构造。

    :param node_feature: 原始输入节点特征 [batch_size, num_nodes, feature_dim]
    :param assignments: 聚类分配矩阵 [batch_size, num_nodes, num_clusters]
    :return: 适应下游任务的图数据 (x, edge_index, batch, edge_attr)
    """

    def construct_node_features(node_feature: torch.Tensor, assignments: torch.Tensor):
        # 将原始特征和聚类分配结果拼接
        node_features = torch.cat([assignments], dim=-1)
        return node_features

    def construct_weighted_graph(node_feature: torch.Tensor):

        adjacency_matrices = node_feature

        return adjacency_matrices

    # 1. 拼接节点特征
    node_features = construct_node_features(node_feature, assignments)

    # 2. 构造加权图的邻接矩阵
    adjacency_matrices = construct_weighted_graph(node_feature)

    # 3. 将邻接矩阵和节点特征转换为图数据
    batch_size, num_nodes, _ = node_features.shape
    x_list = []
    edge_index_list = []
    edge_attr_list = []
    batch_list = []

    for i in range(batch_size):
        edge_index = []
        edge_weight = []

        # 遍历邻接矩阵，生成边的索引和权重
        for j in range(num_nodes):
            for k in range(num_nodes):
                if adjacency_matrices[i, j, k] > 0:  # 保留有权重的边
                    edge_index.append([j, k])
                    edge_weight.append(adjacency_matrices[i, j, k])

        # 转换为 torch tensor
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()  # [2, num_edges]
        edge_weight = torch.tensor(edge_weight, dtype=torch.float)  # [num_edges]

        # 节点特征
        x = node_features[i]  # [num_nodes, feature_dim + num_clusters]

        # 每个样本的 batch 索引，形状为 [num_nodes]
        batch = torch.full((num_nodes,), i, dtype=torch.long)

        # 保存每个图的 x, edge_index, edge_attr, batch 信息
        x_list.append(x)
        edge_index_list.append(edge_index)
        edge_attr_list.append(edge_weight)
        batch_list.append(batch)

    # 将所有样本的数据拼接起来，形成一个批次
    x = torch.cat(x_list, dim=0)  # [total_num_nodes, feature_dim + num_clusters]
    edge_index = torch.cat(edge_index_list, dim=1)  # [2, total_num_edges]
    edge_attr = torch.cat(edge_attr_list, dim=0)  # [total_num_edges]
    batch = torch.cat(batch_list, dim=0)  # [total_num_nodes]

    # 返回适应下游任务的 (x, edge_index, batch, edge_attr)
    return x, edge_index, batch, edge_attr
