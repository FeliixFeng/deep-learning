import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt

# create linear regression sample data
def create_dataset():
    x, y, coef = make_regression(
        n_samples= 100,
        n_features= 1,
        noise= 10,
        coef= True,
        bias= 14.5,
        random_state= 3
    )

    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)

    return x, y, coef

# model train
def train(x, y, coef):
    # 1. create dataset: tensor -> dataset -> dataloader
    dataset = TensorDataset(x, y)

    # 2. create dataloader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 3. initial linear regression model
    model = nn.Linear(1, 1)

    # 4. create loss
    criterion = nn.MSELoss()

    # 5. create optimizer
    optimizer = optim.SGD(model.parameters(), lr = 0.01)

    # 6. concrete train process
    # 6.1 define variable
    epochs, loss_list, total_loss, total_sample = 100, [], 0.0, 0
    # 6.2 begin train in epoch
    for epoch in range(epochs):
        # 6.3 train in batch
        for train_x, train_y in dataloader: # 7 batches(16, 16, ...4)
            # 6.4 model predict
            y_pred = model(train_x)
            # 6.5 compute loss
            loss = criterion(y_pred, train_y.reshape(-1, 1))
            # 6.6 compute total loss
            total_loss += loss.item()
            total_sample += 1
            # grad zero + backward pass + grad update
            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

        # 6.8 append mean loss
        loss_list.append(total_loss / total_sample)
        print(f'epoch: {epoch + 1}, mean loss: {total_loss / total_sample}')

    # 7. print final result
    print(f"{epochs}epochs mean loss: {loss_list}")
    print(f'weight: {model.weight}, bais: {model.bias}')

    # depict loss curve
    plt.figure()
    plt.plot(range(epochs), loss_list)
    plt.title('loss curve change graph')
    plt.grid()
    plt.show()

    # depict the relation between prediction and ground truth
    plt.figure()
    plt.scatter(x, y)

    x_numpy = x.detach().numpy().flatten()
    y_pred = model.weight.item() * x_numpy + model.bias.item()
    y_true = coef * x_numpy + 14.5

    plt.plot(x, y_pred, color='red', label='predictions')
    plt.plot(x, y_true, color='green', label='true')
    plt.legend()
    plt.grid()
    plt.show()














if __name__ == '__main__':
    x, y, coef = create_dataset()
    # print(f'x: {x}, y: {y}, coef: {coef}')
    train(x, y, coef)