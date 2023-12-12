import numpy as np
import string
import time
import torch
import pdb
import math
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

np.random.seed(124)

class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)

        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = 10000**((2 * torch.arange(0, d_model / 2, dtype=torch.float)) / d_model)
        pe[:, 0::2] = torch.sin(position / div_term)
        pe[:, 1::2] = torch.cos(position / div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False)
        return x

pe = PositionalEncoder(512)
input_pe = torch.arange(1, 513)*0.01
input_pe = input_pe.repeat(1, 4, 1).float()

output_pe = pe(input_pe)
print(output_pe.shape)

print(torch.equal(input_pe, output_pe)) 
print(f"input_pe: {input_pe[0, 0, 0:5]} \noutput_pe: {output_pe[0, 0, 0:5]}")
print(f"input_pe: {input_pe[0, 1, 0:5]} \noutput_pe: {output_pe[0, 1, 0:5]}")
print(f"input_pe: {input_pe[0, 2, -5:]} \noutput_pe: {output_pe[0, 2, -5:]}")

import matplotlib.pyplot as plt

p_encode = PositionalEncoder(64)
pe = p_encode.pe.squeeze().T.cpu().numpy()

fig, ax = plt.subplots(2, 2, figsize=(12,4))
dims = [1, 2, 7, 8]
ax = [a for a_list in ax for a in a_list]
for i in range(len(ax)):
    ax[i].plot(np.arange(1,17), pe[dims[i]-1,:16], color=f'C{i}', marker="o", markersize=6, markeredgecolor="black")
    ax[i].set_title(f"Encoding in hidden dimension {dims[i]}")
    ax[i].set_xlabel("Position in sequence", fontsize=10)
    ax[i].set_ylabel("Positional encoding", fontsize=10)
    ax[i].set_xticks(np.arange(1,17))
    ax[i].tick_params(axis='both', which='major', labelsize=10)
    ax[i].tick_params(axis='both', which='minor', labelsize=8)
    ax[i].set_ylim(-1.2, 1.2)
fig.subplots_adjust(hspace=0.8)

plt.show()

def attention(q, k, v):
    d_k = k.size(-1)
    assert d_k == q.size(-1), 'q and k should have the same dimensionality'
    d_v = v.size(-1)

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    att_values = F.softmax(scores, dim=-1)
    output = torch.matmul(att_values, v)

    return output, att_values

torch.manual_seed(42) 
q = torch.randn(2, 5, 512)
k = torch.randn(2, 5, 512)
v = torch.randn(2, 5, 256)

output, att_values = attention(q, k, v)
print(f"Is shape of output correct? {output.shape == v.shape}")
print(f"Is shape of att_values correct? {att_values.shape == torch.Size([q.shape[0], q.shape[1], q.shape[1]])}")
print(f"Do attention values sum to 1? {torch.allclose(torch.sum(att_values, dim=-1), torch.ones(1))}")

print(f"Output check (first): \n{output[0,0,:25]}")
print(f"Output check (last): \n{output[-1,-1,-25:]}")

out = F.scaled_dot_product_attention(q,k,v)
print(f'Is the implementation similar to Pytorch implementation?'
      f' {torch.allclose(output, out, atol=1e-3, rtol=1)}')

class SelfAttention(nn.Module):
    def __init__(self, input_dim, key_dim, output_dim):
        super().__init__()
        self.key_dim = key_dim

        self.W_q = nn.Linear(input_dim, key_dim)
        self.W_k = nn.Linear(input_dim, key_dim)
        self.W_v = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        output, att_weights = attention(q, k, v)

        return output

torch.manual_seed(48)
input_dim = 512
key_dim = 64
output_dim = 512
self_attn = SelfAttention(input_dim, key_dim, output_dim)
x = torch.randn(4, 10, 512)
output = self_attn(x)
print(f"input shape: {x.shape}")
print(f"output shape: {output.shape}")
print(f'Input: \n{x[0,0,:5]}\nOutput: \n{output[0,0,:5]}\n')

random_permutation = torch.randperm(x.size(1))
reverse_permutation = torch.zeros_like(random_permutation)
reverse_permutation[random_permutation] = torch.arange(len(random_permutation))
assert torch.all(x[:, random_permutation, :][:, reverse_permutation, :] == x), 'inverse is incorrect'

x_prime = x[:, random_permutation, :] 
output_prime = self_attn(x_prime)
output_prime_permuted = output_prime[:, reverse_permutation, :]  
print(f'Does the module exhibit permutation-equivaraince?'
      f' {torch.allclose(output, output_prime_permuted, atol=1e-5, rtol=1)}')
print(f'The following two lines should be the same:')
print(output[-1,-1,:10])
print(output_prime_permuted[-1,-1,:10])

class CrossAttention(nn.Module):
    def __init__(self, x_input_dim, y_input_dim, key_dim, output_dim):
        super().__init__()
        self.W_q = nn.Linear(x_input_dim, key_dim)
        self.W_k = nn.Linear(y_input_dim, key_dim)
        self.W_v = nn.Linear(y_input_dim, output_dim)

    def forward(self, x, y):
        q = self.W_q(x)
        k = self.W_k(y)
        v = self.W_v(y)

        output, att_weights = attention(q, k, v)

        return output
    
torch.manual_seed(14)
x_input_dim = 512
y_input_dim = 256
key_dim = 64
output_dim = 128
cross_attn = CrossAttention(x_input_dim, y_input_dim, key_dim, output_dim)
x = torch.randn(3, 10, x_input_dim)
y = torch.randn(3, 10, y_input_dim)
output = cross_attn(x, y)
print(f"input shape x and y: {x.shape}, {y.shape}")
print(f"output shape: {output.shape}")
print(f'x\n{x[0,0,:10]}')
print(f'y\n{y[0,0,:10]}')
print(f'output\n{output[0,0,:10]}')

class MultiHeadedAttention(nn.Module):
    def __init__(self, attn_modules, final_output_dim):
        super().__init__()
        self.attn_modules = nn.ModuleList(attn_modules)
        concatenated_dim = sum([module.W_v.out_features for module in attn_modules])
        self.final_linear = nn.Linear(concatenated_dim, final_output_dim)

    def forward(self, x):
        attn_outputs = []
        for module in self.attn_modules:
            attn_output = module(x)
            attn_outputs.append(attn_output)
        concatenated_outputs = torch.cat(attn_outputs, dim=-1)
        output = self.final_linear(concatenated_outputs)

        return output

torch.manual_seed(10)
input_dim = 256
key_dim = 128
output_dim = 64
final_output_dim = 32
num_heads = 8
attn_modules = [SelfAttention(input_dim, key_dim, output_dim//num_heads) for _ in range(num_heads)]
multi_attn = MultiHeadedAttention(attn_modules, final_output_dim)

x = torch.randn(3, 10, input_dim)
output = multi_attn(x)
print(f"input shape: {x.shape}")
print(f"output shape: {output.shape}")
print(f'x\n{x[0,0,:10]}\noutput\n{output[0,0,:10]}')
