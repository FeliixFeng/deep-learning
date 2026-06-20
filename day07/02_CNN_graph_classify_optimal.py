"""
    CNN Image Classification - Optimized
    CNN图像分类 - 优化版

    Optimizations applied:
    1. Deeper network (3 conv layers instead of 2)
    2. Larger Batch Size (32 instead of 8)
    3. Lower Dropout (0.3 instead of 0.5)
    4. Batch Normalization - accelerate training
"""

import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
from torchsummary import summary

# OPTIMIZATION: Larger batch size (32 instead of 8)
BATCH_SIZE = 32


def create_dataset():
    """
    Load CIFAR10 dataset
    - 50K training images, 10K test images
    - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    - Image size: 32x32x3 (RGB)
    """
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)
    return train_dataset, test_dataset


class ImageModel(nn.Module):
    """
    CNN Architecture (Deeper):
    Input(3,32,32) -> Conv1(32)->BN1->Pool1 -> Conv2(64)->BN2->Pool2 -> Conv3(128)->BN3->Pool3 -> FC1(256) -> FC2(128) -> Output(10)

    Optimizations:
    1. Deeper network: 3 conv layers (32->64->128)
    2. Batch Normalization
    3. Lower Dropout: 0.3
    """
    def __init__(self):
        super().__init__()

        # Conv layers (3 layers, more channels)
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2, 0)

        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2, 0)

        self.conv3 = nn.Conv2d(64, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2, 0)

        # FC layers (lower dropout)
        self.linear1 = nn.Linear(128 * 4 * 4, 256)
        self.dropout1 = nn.Dropout(0.3)  # OPTIMIZATION: 0.3 instead of 0.5
        self.linear2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)  # OPTIMIZATION: 0.3 instead of 0.5
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))

        x = x.reshape(x.size(0), -1)

        x = torch.relu(self.linear1(x))
        x = self.dropout1(x)
        x = torch.relu(self.linear2(x))
        x = self.dropout2(x)

        return self.output(x)


def train(train_dataset):
    """
    Training loop:
    1. Create DataLoader for batching
    2. Initialize model, loss function, optimizer
    3. Train for N epochs
    4. Save model weights
    """
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = ImageModel()

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20
    for epoch in range(epochs):
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()

        for x, y in dataloader:
            model.train()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_correct += (y_pred.argmax(dim=-1) == y).sum()
            total_loss += loss.item() * len(y)
            total_samples += len(y)

        avg_loss = total_loss / total_samples
        accuracy = (total_correct / total_samples) * 100
        elapsed = time.time() - start
        print(f'Epoch {epoch+1}, loss: {avg_loss:.4f}, accuracy: {accuracy:.2f}%, time: {elapsed:.2f}s')

    torch.save(model.state_dict(), './model/image_model.pth')


def evaluate(test_dataset):
    """
    Evaluate model on test set:
    1. Load trained model
    2. Run predictions
    3. Calculate accuracy
    """
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = ImageModel()
    model.load_state_dict(torch.load('./model/image_model.pth'))
    model.eval()

    total_correct, total_samples = 0, 0
    for x, y in dataloader:
        y_pred = torch.argmax(model(x), dim=-1)
        total_correct += (y_pred == y).sum()
        total_samples += len(y)

    accuracy = (total_correct / total_samples) * 100
    print(f'Accuracy: {accuracy:.2f}%')


if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()
    train(train_dataset)
    evaluate(test_dataset)
