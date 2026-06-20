"""
    Case: CNN Image Classification
    演示CNN的综合案例，图像分类

    Review: Deep Learning Project Steps
    1. Prepare dataset
       - Use CIFAR10 from torchvision
       - 60K images (32x32x3), 50K train, 10K test
       - 10 classes, 6K images per class
       - Install: pip install torchvision

    2. Build (Convolutional) Neural Network

    3. Model Training

    4. Model Testing

    Convolutional Layer:
    - Extract local features -> Feature Map
    - Formula: N = (W - F + 2P) // S + 1
    - Each filter is 1 neuron

    Pooling Layer:
    - Dimensionality reduction
    - Max Pooling and Average Pooling
    - Only adjusts H, W; channels remain unchanged
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

BATCH_SIZE = 8

def create_dataset():
    """
    Load CIFAR10 dataset
    - 50K training images, 10K test images
    - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
    - Image size: 32x32x3 (RGB)
    """
    # Load training set
    train_dataset = CIFAR10(root='./data', train=True, transform=ToTensor(), download=True)

    # Load test set
    test_dataset = CIFAR10(root='./data', train=False, transform=ToTensor(), download=True)

    return train_dataset, test_dataset


class ImageModel(nn.Module):
    """
    CNN Architecture for CIFAR10 Classification:
    Input(3, 32, 32) -> Conv1(6) -> Pool1 -> Conv2(16) -> Pool2 -> FC1(120) -> FC2(84) -> Output(10)
    """
    def __init__(self):
        super().__init__()

        # 1. Convolutional layers
        # Conv1: in=3 (RGB), out=6, kernel=3, stride=1, padding=0
        self.conv1 = nn.Conv2d(3, 6, 3, 1, 0)
        # Pool1: kernel=2, stride=2 -> reduce H,W by half
        self.pool1 = nn.MaxPool2d(2, 2, 0)

        # Conv2: in=6, out=16, kernel=3, stride=1, padding=0
        self.conv2 = nn.Conv2d(6, 16, 3, 1, 0)
        # Pool2: kernel=2, stride=2 -> reduce H,W by half
        self.pool2 = nn.MaxPool2d(2, 2, 0)

        # 2. Fully connected layers
        # FC1: 16*6*6=576 -> 120
        self.linear1 = nn.Linear(576, 120)
        # FC2: 120 -> 84
        self.linear2 = nn.Linear(120, 84)
        # Output: 84 -> 10 (10 classes)
        self.output = nn.Linear(84, 10)

    def forward(self, x):
        # Conv1 -> ReLU -> Pool1
        x = self.pool1(torch.relu(self.conv1(x)))

        # Conv2 -> ReLU -> Pool2
        x = self.pool2(torch.relu(self.conv2(x)))

        # Flatten: (batch, 16, 6, 6) -> (batch, 576)
        x = x.reshape(x.size(0), -1)
        # print(f'x.shape: {x.shape}')

        # FC1 -> ReLU
        x = torch.relu(self.linear1(x))

        # FC2 -> ReLU
        x = torch.relu(self.linear2(x))

        # Output layer (no activation, use CrossEntropyLoss)
        return self.output(x)


def train(train_dataset):
    """
    Training loop:
    1. Create DataLoader for batching
    2. Initialize model, loss function, optimizer
    3. Train for N epochs
    4. Save model weights
    """
    # 1. Create DataLoader
    dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Initialize model
    model = ImageModel()

    # 3. Loss function: CrossEntropyLoss (includes softmax)
    criterion = nn.CrossEntropyLoss()

    # 4. Optimizer: Adam with learning rate 0.001
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 5. Training loop
    epochs = 10
    for epoch in range(epochs):
        total_loss, total_samples, total_correct, start = 0.0, 0, 0, time.time()

        for x, y in dataloader:
            model.train()  # Set model to training mode
            y_pred = model(x)  # Forward pass
            loss = criterion(y_pred, y)  # Compute loss
            optimizer.zero_grad()  # Clear gradients
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

            # Compute metrics
            total_correct += (y_pred.argmax(dim=-1) == y).sum()
            total_loss += loss.item() * len(y)
            total_samples += len(y)

        # Print progress
        avg_loss = total_loss / total_samples
        accuracy = (total_correct / total_samples) * 100
        elapsed = time.time() - start
        print(f'Epoch {epoch+1}, loss: {avg_loss:.4f}, accuracy: {accuracy:.2f}%, time: {elapsed:.2f}s')

    # 6. Save model
    torch.save(model.state_dict(), './model/image_model.pth')




def evaluate(test_dataset):
    """
    Evaluate model on test set:
    1. Load trained model
    2. Run predictions
    3. Calculate accuracy
    """
    # 1. Create DataLoader
    dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 2. Load trained model
    model = ImageModel()
    model.load_state_dict(torch.load('./model/image_model.pth'))
    model.eval()  # Set model to evaluation mode

    # 3. Evaluate
    total_correct, total_samples = 0, 0
    for x, y in dataloader:
        y_pred = torch.argmax(model(x), dim=-1)  # Get class with highest probability
        total_correct += (y_pred == y).sum()  # Count correct predictions
        total_samples += len(y)

    # 4. Print accuracy
    accuracy = (total_correct / total_samples) * 100
    print(f'Accuracy: {accuracy:.2f}%')






if __name__ == '__main__':
    train_dataset, test_dataset = create_dataset()
    # model = ImageModel()
    # summary(model, input_size=(3, 32, 32), batch_size=BATCH_SIZE)
    # train(train_dataset)
    evaluate(test_dataset)