"""
From https://github.com/vlukiyanov/pt-dec
"""

import torch
import torch.nn as nn
from typing import Tuple
from .cluster import ClusterAssignment


class DEC(nn.Module):
    def __init__(
        self,
        cluster_number: int,
        hidden_dimension: int,
        encoder: torch.nn.Module,
        alpha: float = 1.0,
        orthogonal=True,
        freeze_center=True, project_assignment=True
    ):
        """
        Module which holds all the moving parts of the DEC algorithm, as described in
        Xie/Girshick/Farhadi; this includes the AutoEncoder stage and the ClusterAssignment stage.

        :param cluster_number: number of clusters
        :param hidden_dimension: hidden dimension, output of the encoder
        :param encoder: encoder to use
        :param alpha: parameter representing the degrees of freedom in the t-distribution, default 1.0
        """
        super(DEC, self).__init__()
        self.encoder = encoder
        self.hidden_dimension = hidden_dimension
        self.cluster_number = cluster_number
        self.alpha = alpha
        self.assignment = ClusterAssignment(
            cluster_number, self.hidden_dimension, alpha, orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment
        )

        self.loss_fn = nn.KLDivLoss(size_average=False)

    def forward(self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the cluster assignment using the ClusterAssignment after running the batch
        through the encoder part of the associated AutoEncoder module.

        :param batch: [batch size, embedding dimension] FloatTensor
        :return: [batch size, number of clusters] FloatTensor 这个方法是低维度中进行聚类操作，让模型在复杂的高纬度当中能够提出出有意义的聚类，因为认为在同一个任务当中，具有相同功能的脑区可能会表现出相同的任务
        """
        node_num = batch.size(1)
        batch_size = batch.size(0)

        # [batch size, embedding dimension] 传过来的batch是bs*200*200的
        flattened_batch = batch.view(batch_size, -1)
        encoded = self.encoder(flattened_batch)  # 这个encoder就是bnt那里面定义的encoder 就算几个线性操作
        # [batch size * node_num, hidden dimension]
        encoded_yuan = encoded
        encoded = encoded.view(batch_size * node_num, -1)
        # [batch size * node_num, cluster_number]
        assignment = self.assignment(encoded) # assignment是聚类的作用；assignment的形状是[batchsize*node_num hidden dimension]
        # [batch size, node_num, cluster_number]
        assignment = assignment.view(batch_size, node_num, -1)
        # [batch size, node_num, hidden dimension]
        encoded = encoded.view(batch_size, node_num, -1)
        # Multiply the encoded vectors by the cluster assignment to get the final node representations
        # [batch size, cluster_number, hidden dimension]  对这两个矩阵进行第二个维度和第三个维度之间相乘 第一个维度并不乘
        # 这么做的目的是将聚类分配（assignment）与编码后的特征向量（encoded）进行矩阵乘法，从而得到最终的节点表示（node_repr）；聚类分配就是每个样本属于哪个簇的概率
        node_repr = torch.bmm(assignment.transpose(1, 2), encoded)
        return node_repr, assignment

    def target_distribution(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
        Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

        :param batch: [batch size, number of clusters] Tensor of dtype float
        :return: [batch size, number of clusters] Tensor of dtype float
        """
        weight = (batch ** 2) / torch.sum(batch, 0)
        return (weight.t() / torch.sum(weight, 1)).t()

    def loss(self, assignment):
        flattened_assignment = assignment.view(-1, assignment.size(-1))
        target = self.target_distribution(flattened_assignment).detach()
        return self.loss_fn(flattened_assignment.log(), target) / flattened_assignment.size(0)

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.assignment.get_cluster_centers()
