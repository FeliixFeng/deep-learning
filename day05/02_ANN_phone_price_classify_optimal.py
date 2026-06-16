"""
    ANN (Artificial Neural Network) for Phone Price Classification - Optimized
    Task: Predict phone price range based on 20 features
    Dataset: train.csv (phone specifications and price ranges)

    Optimizations applied:
    1. Optimizer: SGD -> Adam
    2. Learning rate: 1e-3 -> 1e-4
    3. Data standardization
    4. Deeper network (more parameters)
    5. Adjusted training epochs
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from torchsummary import summary


def create_dataset():
    """
    Load and prepare dataset:
    1. Load CSV data
    2. Split features (X) and labels (Y)
    3. Train/test split (80%/20%)
    4. Standardize features
    5. Convert to PyTorch tensors
    """
    # 1. Load CSV dataset
    data = pd.read_csv('./data/train.csv')

    # 2. Split features and labels
    x, y = data.iloc[:, :-1], data.iloc[:, -1]

    # 3. Convert features to float32 for PyTorch
    x = x.astype(np.float32)

    # 4. Train/test split with stratification
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=3, stratify=y
    )

    # 5. Standardize features (optimization ①)
    transfer = StandardScaler()
    x_train = transfer.fit_transform(x_train)
    x_test = transfer.transform(x_test)

    # 6. Convert to PyTorch tensors and create datasets
    train_dataset = TensorDataset(torch.tensor(x_train), torch.tensor(y_train.values))
    test_dataset = TensorDataset(torch.tensor(x_test), torch.tensor(y_test.values))

    # 7. Return datasets and dimensions
    return train_dataset, test_dataset, x_test.shape[1], len(np.unique(y))


class PhonePriceModel(nn.Module):
    """
    Neural Network Architecture (Deep):
    Input(20) -> Linear(128) -> ReLU -> Linear(256) -> ReLU -> Linear(512) -> ReLU -> Linear(128) -> ReLU -> Linear(output)
    """
    def __init__(self, input_dim, output_dim):
        super(PhonePriceModel, self).__init__()
        # 1. Hidden layer 1: input_dim -> 128
        self.linear1 = nn.Linear(input_dim, 128)
        # 2. Hidden layer 2: 128 -> 256
        self.linear2 = nn.Linear(128, 256)
        # 3. Hidden layer 3: 256 -> 512
        self.linear3 = nn.Linear(256, 512)
        # 4. Hidden layer 4: 512 -> 128
        self.linear4 = nn.Linear(512, 128)
        # 5. Output layer: 128 -> output_dim
        self.linear5 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.linear1(x))  # Layer 1 + ReLU
        x = torch.relu(self.linear2(x))  # Layer 2 + ReLU
        x = torch.relu(self.linear3(x))  # Layer 3 + ReLU
        x = torch.relu(self.linear4(x))  # Layer 4 + ReLU
        x = self.linear5(x)              # Output (no activation, use CrossEntropyLoss)
        return x


def train(train_dataset, input_dim, output_dim):
    """
    Training loop:
    1. Create DataLoader for batching
    2. Initialize model, loss function, optimizer
    3. Train for N epochs
    4. Save model weights
    """
    # Create DataLoader for mini-batch training
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # Initialize model
    model = PhonePriceModel(input_dim, output_dim)

    # Loss function: CrossEntropyLoss (includes softmax)
    criterion = nn.CrossEntropyLoss()

    # Optimizer: Adam (optimization ③), learning rate 1e-4 (optimization ④)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Training loop
    epochs = 50
    for epoch in range(epochs):
        total_loss, batch_num = 0.0, 0
        start = time.time()

        for x, y in train_loader:
            model.train()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_num += 1

        # Print progress every epoch
        avg_loss = total_loss / batch_num
        elapsed = time.time() - start
        print(f"epoch: {epoch+1:02d}, loss: {avg_loss:.4f}, time: {elapsed:.2f}s")

    # Save trained model
    torch.save(model.state_dict(), './model/model.pth')

# todo 4. model test
def evaluate(test_dataset, input_dim, output_dim):
    """
    Evaluate model on test set:
    1. Load trained model
    2. Run predictions
    3. Compare with ground truth
    4. Calculate accuracy
    """
    # 1. Load trained model weights
    model = PhonePriceModel(input_dim, output_dim)
    model.load_state_dict(torch.load('./model/model.pth'))

    # 2. Create test DataLoader
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    # 3. Evaluate
    correct = 0
    for x, y in test_loader:
        model.eval()  # Set model to evaluation mode
        y_pred = model(x)  # Forward pass
        y_pred = torch.argmax(y_pred, dim=1)  # Get class with the highest probability
        correct += (y_pred == y).sum()  # Count correct predictions

    # 4. Print accuracy (percentage)
    accuracy = correct / len(test_dataset) * 100
    print(f"Accuracy: {accuracy:.2f}%")



if __name__ == '__main__':
    # 1. Create dataset
    train_dataset, test_dataset, input_dim, output_dim = create_dataset()

    # 2. Train model (uncomment to run)
    train(train_dataset, input_dim, output_dim)

    # 3. Evaluate model
    evaluate(test_dataset, input_dim, output_dim)