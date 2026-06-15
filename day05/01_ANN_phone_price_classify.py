"""
    1. ANN(Artificial Neural Network)
    base on 20 feature of phone -> predict the price range of phone
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import  torch.optim as optim
from sklearn.model_selection import train_test_split
import  matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time


# todo 1. create dataset
def create_dataset():
    # 1. load csv dataset
    data = pd.read_csv('./data/train.csv')

    # 2. get x feature, y label
    x, y = data.iloc[:, :-1], data.iloc[:,-1]

    # 3. turn feature into float
    x = x.astype(np.float32)

    # 4. split train set and test set
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=3, stratify=y
    )

    # 5. turn dataset into tensor
    train_dataset = TensorDataset(torch.tensor(x_train.values), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test.values), torch.tensor(y_test.values))

    # 6. return result
    return train_dataset, test_dataset, x_test.shape[1], len(np.unique(y))

# todo 2. create neural network
class PhonePriceModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, 128)
        self.linear2 = nn.Linear(128, 256)
        self.output = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = self.output(x)
        return x

# todo 3. model train

# todo 4. model test


if __name__ == '__main__':
    train_dataset, test_dataset, input_dim, output_dim =  create_dataset()
    # print(f'train set: {train_dataset}')
    # print(f'test set: {test_dataset}')
    # print(f'input feature: {input_dim}')
    # print(f'output label: {output_dim}')

    models