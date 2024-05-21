import torch
import torch.nn.functional as F

# Define the matrices
Q = torch.tensor([
    [0.5, 0.2, 0.24, 0.46],
    [0.13, 0.31, 0.25, 0.04],
    [0.4, 0.15, 0.24, 0.56]
], dtype=torch.float32)

K = torch.tensor([
    [0.44, 0.15, 0.21, 0.56],
    [0.67, 0.75, 0.7, 0.59],
    [0.18, 0.23, 0.23, 0.17]
], dtype=torch.float32)

V = torch.tensor([
    [0.29, 0.56, 0.47, 0.15],
    [0.21, 0.07, 0.1, 0.26],
    [0.54, 0.31, 0.35, 0.61]
], dtype=torch.float32)

# Compute the dot product of Q and the transpose of K
QK_T = torch.matmul(Q, K.T)

# Scale by the square root of the dimensionality of the key vectors (number of columns in K)
d_k = K.size(1)
QK_T_scaled = QK_T / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

# Apply softmax to get the attention weights
attention_weights = F.softmax(QK_T_scaled, dim=-1)

# Multiply the attention weights by the value matrix V to get the output
output = torch.matmul(attention_weights, V)

# Print the results
print("Attention Weights:\n", attention_weights)
print("Output:\n", output)

