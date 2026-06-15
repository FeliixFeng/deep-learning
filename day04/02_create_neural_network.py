import torch
import torch.nn as nn
from torchsummary import summary

# 1. create neural network nn.Module
class ModelDemo(nn.Module):
    # todo: 1. initial
    def __init__(self):
        # 1.1 super
        super().__init__()
        # 1.2 create neural network -> hidden layer + output layer
        # hidden layer1: input: 3, output: 3, activation function: Sigmoid
        self.linear1 = nn.Linear(3, 3)
        # hidden layer2: input: 3, output: 2
        self.linear2 = nn.Linear(3, 2)
        # output layer: input: 2, output: 2
        self.output = nn.Linear(2, 2)

        # 1.3 initial hidden layer
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)

        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)

    # todo: 2. forward: input layer -> hidden layer -> output layer
    def forward(self, x):
        # 2.1 hidden layer1 compute: weighted sum, activation function
        x = torch.sigmoid(self.linear1(x))

        # 2.2 hidden layer2 compute
        x = torch.relu(self.linear2(x))

        # 2.3 output layer compute
        x = torch.softmax(self.output(x), dim=1)

        # 2.4 return prediction
        return x

# 2. model train
def train():
    # 1. create model
    my_model = ModelDemo()
    # print(f'my_model: {my_model}')

    # 2. create dataset
    data = torch.randn(size=(5, 3))
    print(f'data: {data}')
    print((f'data.shape: {data.shape}'))
    print((f'data.requires_grad: {data.requires_grad}'))
    
    # 3. training model
    output = my_model(data)
    print(f'output: {output}')
    print((f'output.shape: {output.shape}'))
    print((f'data.requires_grad: {output.requires_grad}'))
    print('-'* 30)

    # 4. compute and print model parameter
    print('=====================compute model parameter=====================')
    # p1: model class, p2: dataset dimension
    summary(my_model, input_size=(5, 3))

    print('=====================view model parameter=====================')
    for name, param in my_model.named_parameters():
        print(f'name: {name}')
        print(f'param: {param} \n')






if __name__ == '__main__':
    train()