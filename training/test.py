import torch
# Hypothetical parameters
batch_size = 16
num_nodes = 116  # Number of nodes in each sample
num_clusters = 10  # Number of clusters

# Randomly generate assignment_static
# Generate a tensor of size [batch_size, num_nodes, num_clusters]
assignment_static = torch.zeros(batch_size, num_nodes, num_clusters)
log_static_assignment = torch.log_softmax(assignment_static, dim=-1)
c = 1