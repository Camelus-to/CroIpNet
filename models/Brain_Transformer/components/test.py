# 假设 all_PL_features 的形状是 [batch_size, time_steps, feature_dim]，即 [16, 6, 1000]
# time_weights 的形状是 [batch_size, time_steps, num_nodes]，即 [16, 6, 116]
import torch
time_weights = torch.randn(16, 6, 116)
all_PL_features = torch.randn(16, 6, 1000)
# 扩展 time_weights 的维度，使其可以与 all_PL_features 进行加权
# time_weights 的形状是 [16, 6, 116]，我们需要扩展为 [16, 6, 116, 1] 以便进行加权
time_weights_expanded = time_weights.unsqueeze(-1)  # [16, 6, 116] -> [16, 6, 116, 1]

# all_PL_features 的形状是 [16, 6, 1000]，我们需要在时间维度上进行加权
# 扩展 all_PL_features 的维度，使其能够与 time_weights_expanded 进行逐元素相乘
# 所以我们将 all_PL_features 扩展到 [16, 6, 1, 1000]，然后与 time_weights_expanded 相乘
all_PL_features_expanded = all_PL_features.unsqueeze(2)  # [16, 6, 1, 1000]

# 进行加权操作
weighted_PL_features = all_PL_features_expanded * time_weights_expanded  # [16, 6, 116, 1000]

# 对时间维度进行汇总，按照节点维度对加权后的特征求和
# 对于每个样本，我们希望得到一个 [feature_dim] 的特征，所以在时间维度上对每个加权特征求和
# 求和维度是 `dim=1`，即对每个样本的时间维度求和
weighted_PL_features_sum = weighted_PL_features.sum(dim=1)  # [16, 116, 1000]

# 最终，你希望得到 [batch_size, feature_dim]，即 [16, 1000]，这时需要对节点维度求和或平均
final_result = weighted_PL_features_sum.sum(dim=1)  # [16, 1000]

print(final_result.shape)  # 输出 [16, 1000]  # [16, 6, 1000] 求和，针对每个节点的加权进行合并
s = 1