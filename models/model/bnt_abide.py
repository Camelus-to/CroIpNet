from omegaconf import DictConfig
from ..base import BaseModel
from torch.nn import Linear
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.functional as F


class GCNPredictor(nn.Module):
    def __init__(self, node_input_dim=200, roi_num=200):
        super().__init__()
        inner_dim = roi_num
        self.roi_num = roi_num
        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(inner_dim, inner_dim)
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)

        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)

        self.fcn = nn.Sequential(
            nn.Linear(int(8 * int(roi_num * 0.7)), 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )
        self.norm = torch.nn.LayerNorm(normalized_shape=roi_num, elementwise_affine=True)
        self.weight = torch.nn.Parameter(torch.Tensor(1, 8))

        self.softmax = nn.Sigmoid()

    # node是加上偏差后的功能连接矩阵 知识图是偏相关的 data_graph是经过时间编码
    def forward(self, node_feature, knowledge_graph, data_graph):
        bz = data_graph.shape[0]  # batch

        h1 = node_feature
        h2 = data_graph
        x = h1 * h2

        x = self.gcn2(x)

        x = self.bn3(x)

        return x  # 返回的x是batch*116*8的


import torch
import torch.nn as nn


class MultiScaleTimeSeriesModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, lstm_layers=1, fusion_size=128,
                 transformer_d_model=116, transformer_nhead=4, transformer_layers=2, dropout=0.1):
        super(MultiScaleTimeSeriesModel, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size

        # LSTM for multi-scale
        self.lstm_scale1 = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)
        self.lstm_scale2 = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)
        self.lstm_scale3 = nn.LSTM(input_size, hidden_size, lstm_layers, batch_first=True)

        # Fusion layer for LSTM features
        self.fusion_linear = nn.Linear(hidden_size * 3, fusion_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=transformer_d_model, nhead=transformer_nhead,
                                                   dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # Output layer
        self.out_linear = nn.Linear(transformer_d_model, 1)
        self.relu = nn.ReLU()

        # Project fused features to transformer input dimension
        self.proj_linear = nn.Linear(fusion_size, transformer_d_model)

    def forward(self, x):
        """
        x shape: [batch_size, num_nodes, seq_length]
        """
        batch_size, num_nodes, seq_length = x.shape
        x = x.unsqueeze(-1)  # Shape: [batch_size, num_nodes, seq_length, 1]

        # Process each node separately in multi-scale LSTM
        x_scale1 = x  # Original scale
        x_scale2 = x[:, :, ::2, :]  # Downsample by 2
        x_scale3 = x[:, :, ::4, :]  # Downsample by 4

        # Reshape for LSTM: [batch*num_nodes, seq_length, 1]
        x_scale1 = x_scale1.reshape(-1, seq_length, 1)
        x_scale2 = x_scale2.reshape(-1, x_scale2.shape[2], 1)
        x_scale3 = x_scale3.reshape(-1, x_scale3.shape[2], 1)

        # Pass through LSTM
        _, (h_n1, _) = self.lstm_scale1(x_scale1)
        _, (h_n2, _) = self.lstm_scale2(x_scale2)
        _, (h_n3, _) = self.lstm_scale3(x_scale3)

        # Concatenate LSTM features
        fused_feature = torch.cat([h_n1[-1], h_n2[-1], h_n3[-1]], dim=-1)  # Shape: [batch*num_nodes, hidden_size*3]
        fused_feature = self.relu(self.fusion_linear(fused_feature))  # Shape: [batch*num_nodes, fusion_size]

        # Reshape for Transformer: [batch_size, num_nodes, fusion_size]
        fused_feature = fused_feature.view(batch_size, num_nodes, -1)

        # Prepare for Transformer: [num_nodes, batch_size, fusion_size]
        transformer_input = self.proj_linear(fused_feature).transpose(0, 1)

        # Transformer encoder
        transformer_output = self.transformer_encoder(
            transformer_input)  # Shape: [num_nodes, batch_size, d_model] 是116*8*116的 在下面只取了最后一个节点的特征
        data_graph = transformer_output.transpose(0, 1)
        # Use the final node's feature (or apply other aggregation)
        final_feature = transformer_output[-1, :, :]  # Shape: [batch_size, d_model] final_feature是16*128的

        # Output layer
        out = self.out_linear(final_feature).squeeze(-1)  # Shape: [batch_size]

        return out, transformer_output, data_graph


class Individual(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.node_feature_dim = 116
        self.node_feature_dim_dynamic = 116
        self.predictor = GCNPredictor(self.node_feature_dim, roi_num=config.node_sz)
        self.fc_p = nn.Sequential(nn.Linear(in_features=config.node_sz, out_features=config.node_sz),
                                  nn.LeakyReLU(negative_slope=0.2))
        # self.time_attention = TimeAttention(input_dim=116)
        self.num_time_length = config.num_time_length
        self.lstm_transformer = MultiScaleTimeSeriesModel().to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def contrastive_loss(self, features, labels, margin=1.0):

        batch_size, num_nodes, feature_dim = features.shape
        features_flat = features.view(batch_size, -1)
        # 计算余弦相似度
        cosine_sim = F.cosine_similarity(features_flat.unsqueeze(1), features_flat.unsqueeze(0), dim=-1)
        # 获取标签信息：同类别标签为1，不同类别标签为0
        labels = labels.argmax(dim=1)
        labels = labels.unsqueeze(1) == labels.unsqueeze(0)  # 标签相同则为True，反之为False
        labels = labels.float()  # 将标签转换为浮点型，方便与相似度做计算

        # 计算损失
        loss = 0.0
        n = features.size(0)

        for i in range(n):
            for j in range(n):
                if i != j:  # 跳过自身
                    if labels[i, j]:
                        # 同类别样本，期望相似度较高
                        loss += (1 - cosine_sim[i, j])  # 越相似损失越小
                    else:
                        # 不同类别样本，期望相似度较低
                        loss += torch.max(torch.tensor(0.0), margin - cosine_sim[i, j])  # 保证最小相似度

        loss = loss / (n * (n - 1))  # 对样本对进行平均
        return loss

    def forward(self, final_pearson, time_series, label,
                pseudo):
        batch_size, num_nodes, time_steps = time_series.shape
        outputs, _, fc_matrices = self.lstm_transformer(time_series)
        # fc_matrices = torch.tensor(fc_matrices)
        # fc_matrices = fc_matrices.to(torch.float32)
        data_graph = self.emb2graph(fc_matrices).to(torch.float32)
        bz, _, _ = data_graph.shape
        edge_variance = torch.mean(torch.var(data_graph.reshape((bz, -1)), dim=1))
        pseudo = self.fc_p(pseudo)
        nodes = final_pearson + pseudo
        #
        feature_I = self.predictor(nodes, data_graph, data_graph)
        contrastive_loss_label = self.contrastive_loss(feature_I, label)
        return feature_I, data_graph, contrastive_loss_label


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.projection_head = nn.Sequential(
            nn.Linear(116 * 116, 116),
            nn.ReLU(),
            nn.Linear(116, 116)
        )

    def forward(self, features, labels):
        # 展平每个特征矩阵
        features_flat = features.view(features.size(0), -1)  # [batch_size, feature_dim * feature_dim]
        projected_features = self.projection_head(features_flat)  # [batch_size, projection_dim]
        features_flat = projected_features / projected_features.norm(dim=1, keepdim=True)
        # 计算相似度矩阵
        sim_matrix = torch.matmul(features_flat, features_flat.T) / self.temperature

        # 计算正负样本对
        labels = labels.argmax(dim=1)
        labels = labels.unsqueeze(1)
        mask = labels == labels.T

        # 获取正样本和负样本的索引
        pos_sim = sim_matrix[mask]
        neg_sim = sim_matrix[~mask]

        # 计算正样本和负样本的数量
        pos_count = pos_sim.size(0)
        neg_count = neg_sim.size(0)

        if pos_count != neg_count:
            min_count = min(pos_count, neg_count)
            pos_sim = pos_sim[:min_count]
            neg_sim = neg_sim[:min_count]

        # 对比损失（InfoNCE Loss）
        loss_pos = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.exp(neg_sim)))
        loss_pos_mean = loss_pos.mean()  # 计算平均损失
        sim_matrix_min = sim_matrix.min()
        sim_matrix_max = sim_matrix.max()

        sim_matrix = (sim_matrix - sim_matrix_min) / (sim_matrix_max - sim_matrix_min)
        return loss_pos_mean, sim_matrix  # 返回损失和相似度矩阵


class Population(nn.Module):
    def __init__(self, roi_num=116, node_input_dim=116):
        super(Population, self).__init__()

        inner_dim = roi_num

        self.roi_num = roi_num
        self.node_input_dim = node_input_dim
        self.inner_dim = 116
        self.fc_p = nn.Sequential(nn.Linear(in_features=roi_num, out_features=roi_num),
                                  nn.LeakyReLU(negative_slope=0.2))

        self.contrastive_loss = ContrastiveLoss()

        self.gcn = nn.Sequential(
            nn.Linear(node_input_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
            Linear(inner_dim, inner_dim)
        )
        self.bn1 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn1 = nn.Sequential(
            nn.Linear(inner_dim, inner_dim),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn2 = torch.nn.BatchNorm1d(inner_dim)
        self.gcn2 = nn.Sequential(
            nn.Linear(inner_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 8),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.bn3 = torch.nn.BatchNorm1d(inner_dim)
        self.fcn = nn.Sequential(
            nn.Linear(int(8 * int(roi_num * 0.7)), 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, 2)
        )
        self.weight = torch.nn.Parameter(torch.Tensor(1, 8))
        self.softmax = nn.Sigmoid()

    def forward(self, x, pseudo, ages, genders, site):
        self.bz = x.shape[0]
        bz = x.shape[0]
        pseudo = self.fc_p(pseudo)
        x = x + pseudo
        topk = 8

        contrastive_loss, sim_matrix = self.contrastive_loss(x, site)

        a = 0.9
        beta = 2
        alpha = 0.1
        num_nodes = x.shape[0]
        dist = np.zeros((num_nodes, num_nodes))
        inds = []

        for i in range(num_nodes):
            vector1 = x[i].flatten()
            for j in range(num_nodes):
                vector2 = x[j].flatten()

                feature_similarity = np.exp(
                    -alpha * (np.linalg.norm(vector1.cpu().detach().numpy() - vector2.cpu().detach().numpy())))

                contrastive_similarity = sim_matrix[i, j].cpu().detach().numpy()

                total_similarity = feature_similarity * (1 - contrastive_similarity) + contrastive_similarity

                if abs(ages[i] - ages[j]) <= beta and genders[i] == genders[j]:
                    dist[i][j] = total_similarity * a
                else:
                    dist[i][j] = total_similarity * (1 - a)

            ind = np.argpartition(dist[i, :], -topk)[-topk:]
            inds.append(ind)

        adj = np.zeros((num_nodes, num_nodes))
        for i, pos in enumerate(inds):
            for j in pos:
                adj[i][j] = dist[i][j]

        adj = torch.tensor(adj).cuda()
        adj = adj.to(torch.float)

        x = x.reshape((bz, -1))
        x = adj @ x
        x = x.reshape((bz, self.roi_num, -1))

        x = self.gcn(x)

        x = x.reshape((bz * self.roi_num, -1))
        x = self.bn1(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = x.reshape((bz, -1))
        x = adj @ x
        x = x.reshape((bz, self.roi_num, -1))

        x = self.gcn1(x)

        x = x.reshape((bz * self.roi_num, -1))
        x = self.bn2(x)
        x = x.reshape((bz, self.roi_num, -1))

        x = x.reshape((bz, -1))
        x = adj @ x
        x = x.reshape((bz, self.roi_num, -1))
        x = self.gcn2(x)

        x = self.bn3(x)

        return x, contrastive_loss


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, feature_I, feature_P):
        Q = self.query(feature_I)
        K = self.key(feature_P)
        V = self.value(feature_P)

        # 计算注意力权重
        attention = self.softmax(torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.shape[-1]))
        # 通过注意力机制加权
        fused_feature = torch.matmul(attention, V)

        return fused_feature


class ImprovedFeatureFusion(nn.Module):
    def __init__(self, feature_dim=8):
        super().__init__()
        self.attention = nn.MultiheadAttention(feature_dim, num_heads=4)
        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, individual_feat, population_feat):
        attn_output, _ = self.attention(
            individual_feat,
            population_feat,
            population_feat
        )

        gate = self.gate(torch.cat([individual_feat, population_feat], dim=-1))
        fused_features = gate * individual_feat + (1 - gate) * population_feat

        return fused_features


class BrainNetworkTransformer_abide(BaseModel):

    def __init__(self, config: DictConfig, semantic_similarity_matrix):
        super().__init__()
        self.node_feature_dim_dynamic = 116
        self.node_size = 116
        self.node_feature_size = 116
        self.out_dim = 2
        self.model_I = Individual(config.dataset)  # corr[0] corr[1] t[1]
        self.model_P = Population(self.node_size, self.node_feature_size)
        self.lam_I = 1
        self.lam_P = 1
        self.weight = torch.nn.Parameter(torch.Tensor(1, 8))
        self.softmax = nn.Sigmoid()
        self.fcn = nn.Sequential(
            nn.Linear(int(8 * int(self.node_size)), 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, self.out_dim)
        )
        self.cross_attention = CrossAttentionFusion(dim=8)
        self.feature_fusion = ImprovedFeatureFusion()
        self.fusion_layer = nn.Sequential(
            nn.Linear(int(16 * int(self.node_size)), 256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(256, 32),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(32, self.out_dim)
        )

    def forward(self,
                final_pearson, signals, pseudo, ages, genders, site, label):
        final_pearson[torch.isnan(final_pearson)] = 0
        signals[torch.isnan(signals)] = 0  # 该模型的输入 node_feature是200*200的，就是一个功能连接矩阵，而 time_seires是200*100的，是时间序列
        bz, num_nodes, time_steps = signals.shape

        feature_I, data_graph, contrastive_loss_label = self.model_I(final_pearson, signals, label, pseudo)

        feature_P, contrastive_loss = self.model_P(final_pearson, pseudo, ages, genders, site)
        # 再添加一个图正则化损失
        bz = final_pearson.shape[0]
        fused_feature = self.cross_attention(feature_I, feature_P)  # 拼接特征

        # 使用门控机制
        x = self.feature_fusion(feature_I, feature_P).cuda()
        x = torch.cat([feature_I, feature_P], dim=-1)  # 拼接特征
        x = x.view(bz, -1).cuda()
        out = self.fusion_layer(x)  # 映射到统一维度
        # out = self.fcn(fused_feature)
        return [out, out], data_graph, contrastive_loss_label, contrastive_loss
