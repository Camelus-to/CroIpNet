from torch_geometric.nn import GATConv
import torch.nn as nn
import torch
import torch.nn.functional as F
import os


# 计算特征表示的相似度
def compute_similarity(features1, features2):
    features1 = F.normalize(features1, dim=-1)
    features2 = F.normalize(features2, dim=-1)
    return torch.matmul(features1, features2.transpose(-1, -2))  # [batch_size, num_nodes, num_nodes]


# 生成对比学习的损失
def contrastive_loss(similarity_matrix):
    batch_size, num_nodes, _ = similarity_matrix.shape
    labels = torch.arange(num_nodes).unsqueeze(0).expand(batch_size, -1).to(similarity_matrix.device)
    loss = F.cross_entropy(similarity_matrix, labels)
    return loss


def identify_abnormal_clusters(assignments_static, assignments_dynamic):
    # 获取异常节点
    low_consistency_nodes = calculate_consistency(assignments_static, assignments_dynamic)

    # 统计每个簇的异常节点数量
    abnormal_clusters = {}
    for node in low_consistency_nodes:
        cluster = assignments_static[node].argmax()  # 用静态分配的簇作为基准
        if cluster in abnormal_clusters:
            abnormal_clusters[cluster] += 1
        else:
            abnormal_clusters[cluster] = 1

    return abnormal_clusters  # 返回包含异常节点的簇


def calculate_consistency(assignments_static, assignments_dynamic, threshold=0.5):
    # assignments_static 和 assignments_dynamic 的形状为 [num_nodes, num_clusters]

    num_nodes = assignments_static.shape[0]
    consistency_scores = []

    for node in range(num_nodes):
        # 静态和动态分配中该节点的簇分配结果
        static_cluster = assignments_static[node].argmax()
        dynamic_cluster = assignments_dynamic[node].argmax()

        # 检查节点在不同时间步中是否一致
        consistency = 1 if static_cluster == dynamic_cluster else 0
        consistency_scores.append(consistency)

    # 计算一致性评分，低于阈值的标记为异常
    low_consistency_nodes = [node for node, score in enumerate(consistency_scores) if score < threshold]

    return low_consistency_nodes  # 返回异常节点列表


class GATModel(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, threshold_percent=0.1):
        super(GATModel, self).__init__()
        self.threshold_percent = threshold_percent  # 阈值百分比
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index, edge_weight):
        num_edges = edge_weight.size(0)
        k = int(num_edges * self.threshold_percent)  # 保留前10%的边

        # Step 2: 获取边权重的阈值
        threshold_value = torch.topk(edge_weight, k, largest=True).values.min()

        # Step 3: 根据阈值筛选边
        mask = edge_weight >= threshold_value
        filtered_edge_index = edge_index[:, mask]  # 只保留权重高于阈值的边
        filtered_edge_weight = edge_weight[mask]  # 保留对应的边权重

        x = self.conv1(x, filtered_edge_index, filtered_edge_weight)
        x = self.relu(x)
        x = self.conv2(x, filtered_edge_index, filtered_edge_weight)
        return x


class PositionalEncodingMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(PositionalEncodingMLP, self).__init__()
        # 定义 MLP 层，用于处理位置编码
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, pos_emb):
        # 输入是位置编码，输出的维度与输入一致
        return self.mlp(pos_emb)


class PHAutoEncoder(nn.Module):
    def __init__(self, firland, midland, lasland):
        super(PHAutoEncoder, self).__init__()
        self.encode0 = nn.Linear(firland, midland)
        self.encode1 = nn.Linear(midland, lasland)
        self.encode2 = nn.Linear(midland, lasland)  # 用于计算 logvar
        self.decode0 = nn.Linear(lasland, midland)
        self.decode1 = nn.Linear(midland, firland)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, features):
        embedding = F.leaky_relu(self.encode0(features))  # 经过 Leaky ReLU 激活函数的结果
        mu = self.encode1(embedding)  # 得到的潜在空间表示的均值和对数方差 是均值向量 并不是单纯的一个特征的标量
        logvar = self.encode2(embedding)
        embedding_out = self.reparameterize(mu, logvar)  # 通过重参数化技巧得到的最终潜在空间表示
        embedding_res = F.leaky_relu(self.decode0(embedding_out))  # embedding_res 是通过解码器将潜在空间表示重构回原始特征空间的结果。
        embedding_res = torch.sigmoid(
            self.decode1(embedding_res))  # mu是100*16, logvar是100*16, embedding_out是100*16, embedding_res是100*512
        return mu, logvar, embedding_out, embedding_res


def distance_win_loss(land_group):
    Within_Group = torch.norm(land_group[:, None] - land_group, dim=2, p=2) ** 2
    pho = Within_Group.shape[0] * (Within_Group.shape[0] - 1) / 2
    dis_win_loss = torch.sum(torch.triu(Within_Group, diagonal=1)) / pho
    return dis_win_loss


def distance_wout_loss(cn_group, pat_group):
    Without_Group = torch.norm(cn_group[:, None] - pat_group, dim=2, p=2) ** 2
    dis_wout_loss = torch.mean(Without_Group)
    return dis_wout_loss


def reconstruct_loss(recon_x, x, mu, logvar):
    recons_loss = F.mse_loss(recon_x, x)
    kld_loss = torch.mean(-0.5 * torch.sum(
        1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    return recons_loss + kld_loss


# la_mu 是均值 la_logvar是方差 land_original是原始表示， la_decoder 是通过VAE得到的
def land_loss(land_embed, la_mu, la_logvar, land_original, la_decoder, y, dis=True, pear=True):
    if land_embed[0].shape[0] != y.shape[0]:
        print("Warning: the number of subjects is incorrect in landscapes")
    else:
        L_sub_total = 0
        L_con_total = 0
        for t in range(land_original.shape[1]):  # 这里假设时间步维度是第2维，即6
            # 提取当前时间点的数据
            land_original_t = land_original[:, t, :]  # 当前时间步的原始数据
            la_mu_t = la_mu[t]  # 当前时间步的潜在均值
            la_logvar_t = la_logvar[t]  # 当前时间步的潜在对数方差
            la_decoder_t = la_decoder[t]
            # 计算当前时间步的群内距离损失
            Sub_cn_t = la_mu_t[(y[:, 0] == 0).nonzero().flatten(), :]  # 类别0的数据
            Sub_pat_t = la_mu_t[(y[:, 1] == 1).nonzero().flatten(), :]  # 类别1的数据
            L_sub_t = distance_win_loss(Sub_cn_t) + distance_win_loss(Sub_pat_t)  # 同一类别内部数据点之间的距离损失

            # 计算当前时间步的重建损失
            L_con_t = reconstruct_loss(la_decoder_t, land_original_t, la_mu_t, la_logvar_t)

            # 累加每个时间步的损失
            L_sub_total += L_sub_t
            L_con_total += L_con_t

        # 对所有时间步的损失求平均
        L_sub_avg = L_sub_total / land_original.shape[1]  # 平均群内距离损失
        L_con_avg = L_con_total / land_original.shape[1]  # 平均重建损失
    return L_sub_avg, L_con_avg


# 群内损失 (L_sub)：希望每个类别的样本在潜在空间中更紧密地聚集。
# 群间损失 (L_group)：希望不同类别的样本在潜在空间中距离更远。 学习如何通过 Betti 曲线嵌入将不同类别的样本在潜在空间中正确地聚类。
def BettiCurve_loss(betti, y):
    for t in range(len(betti)):  # 这里假设时间步维度是第2维，即6
        land_original_t = betti[t]
        Sub_cn = land_original_t[(y[:, 0] == 0).nonzero().flatten(), :]
        Sub_pat = land_original_t[(y[:, 1] == 1).nonzero().flatten(), :]
        L_sub = distance_win_loss(Sub_cn) + distance_win_loss(Sub_pat)
        L_group = -distance_wout_loss(Sub_cn, Sub_pat)
    return L_sub + 0.33 * L_group
