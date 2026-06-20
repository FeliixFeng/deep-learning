"""
    RNN (Recurrent Neural Network) Basic Example
    - Process sequential data
    - Maintains hidden state across time steps
"""

import torch
import torch.nn as nn


# Create RNN layer
# input_size=128: feature size of each time step
# hidden_size=256: size of hidden state
# num_layers=1: number of stacked RNN layers
rnn = nn.RNN(input_size=128, hidden_size=256, num_layers=1)

# Input tensor: (seq_len, batch_size, input_size)
# seq_len=5: 5 time steps
# batch_size=32: 32 samples
# input_size=128: 128 features per time step
x = torch.randn(size=(5, 32, 128))

# Initial hidden state: (num_layers, batch_size, hidden_size)
h0 = torch.randn(size=(1, 32, 256))

# Forward pass
# output: (seq_len, batch_size, hidden_size) - all hidden states
# h1: (num_layers, batch_size, hidden_size) - last hidden state
output, h1 = rnn(x, h0)

print(f'Input shape: {x.shape}')      # (5, 32, 128)
print(f'Output shape: {output.shape}')  # (5, 32, 256)
print(f'h1 shape: {h1.shape}')          # (1, 32, 256)
