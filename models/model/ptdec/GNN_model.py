import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from .gnn_conv import GCNConv
from torch.nn import Dropout, Linear, BatchNorm1d


class PHAutoEncoder(nn.Module):
    def __init__(self, firland, midland, lasland):
        super(PHAutoEncoder, self).__init__()
        self.encode0 = nn.Linear(firland, midland)
        self.encode1 = nn.Linear(midland, lasland)
        self.decode0 = nn.Linear(lasland, midland)
        self.decode1 = nn.Linear(midland, firland)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, features):
        embedding = F.leaky_relu(self.encode0(features)) # 经过 Leaky ReLU 激活函数的结果
        mu = self.encode1(embedding) # 得到的潜在空间表示的均值和对数方差
        logvar = self.encode1(embedding)
        embedding_out = self.reparameterize(mu, logvar) # 通过重参数化技巧得到的最终潜在空间表示
        embedding_res = F.leaky_relu(self.decode0(embedding_out)) # embedding_res 是通过解码器将潜在空间表示重构回原始特征空间的结果。
        embedding_res = torch.sigmoid(self.decode1(embedding_res)) # mu是100*16, logvar是100*16, embedding_out是100*16, embedding_res是100*512
        return mu, logvar, embedding_out, embedding_res


class GcnMlp(nn.Module):
    def __init__(self, in_dim, mid_dim, las_dim, dropout):
        super(GcnMlp, self).__init__()
        self.fc1 = Linear(in_dim, mid_dim)
        self.fc2 = Linear(mid_dim, las_dim)
        self.Act1 = nn.ReLU()
        self.Act2 = nn.ReLU()
        self.reset_parameters()
        self.dropout = dropout
        self.BNorm0 = BatchNorm1d(in_dim, eps=1e-5)
        self.BNorm1 = BatchNorm1d(mid_dim, eps=1e-5)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-5)
        nn.init.normal_(self.fc2.bias, std=1e-5)

    def forward(self, x):
        x = self.Act1(self.fc1(self.BNorm0(x))) # 从208变成了256
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.Act2(self.fc2(self.BNorm1(x))) # 从256变成了32
        return x


def check_for_nans_and_infs(tensor, name):
    if torch.isnan(tensor).any():
        print(f"NaNs found in {name}")
    if torch.isinf(tensor).any():
        print(f"Infs found in {name}")


class MyGCN(torch.nn.Module):
    def __init__(self, infeat, nclass, nROI, nhid=32, dropout=0.8,
                 weight_decay=5e-4,
                 with_relu=True,
                 ):
        super(MyGCN, self).__init__()
        self.infeat = infeat
        self.nclass = nclass
        self.dropout = dropout
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        # MLP layer
        self.mid = 256
        self.last = 32
        # Graph Convolutational layer
        self.nhid = 64
        self.outfeat = 16  # outfeat
        self.midcurve = 512
        self.lascurve = 32
        self.nROI = nROI

        self.conv1 = GCNConv(self.infeat, self.nhid, bias=True, normalize=False)
        self.conv2 = GCNConv(self.nhid, self.outfeat, bias=True, normalize=False)
        self.mlp = GcnMlp(self.nhid * 2 + self.outfeat * 2, self.mid, self.last,dropout)  # + self.lasland * 2 + self.lasland
        self.classifier = Linear(self.last, nclass)

    def forward(self, x, edge_index, batch, edge_attr, PH1_feat=None, PH0_feat=None):

        edge_attr = torch.where(torch.isnan(edge_attr), torch.tensor(0.0, device=edge_attr.device), edge_attr)
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # 一个最大池化 一个平均池化
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x_feat = torch.cat([x1, x2], dim=1) # x1是100*128的 x2是100*32的 outfeat_PH是100*48的 又连接了起来 成了100*218的
        x_feat = self.mlp(x_feat)
        classfication = F.log_softmax(self.classifier(x_feat), dim=1)  # 最后得到的也是一个batch_size*2的

        return classfication
