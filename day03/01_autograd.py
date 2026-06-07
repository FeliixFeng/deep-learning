"""
    Automatic Differentiation (autograd):
    1. requires_grad - enable gradient tracking
    2. grad - compute gradients
    3. backward() - backpropagation
"""

import torch


def dm01():
    """
    Basic gradient descent:
    W_new = W_old - lr * grad
    loss = 2w^2 -> grad = 4w
    """
    # w_new = w_old - lr * grad
    # grad = d(loss)/dw
    w = torch.tensor(10.0, requires_grad=True)

    # Define loss
    loss = 2 * w ** 2  # 2w^2 -> grad = 4w

    # Compute gradient (must be scalar)
    loss.backward()

    # Update weight: w = w - 0.01 * grad
    w.data = w.data - 0.01 * w.grad
    print(f"Updated w: {w}")


def dm02():
    """
    Gradient descent loop:
    minimize y = w^2 + 20, optimal w = 0
    """
    w = torch.tensor(10.0, requires_grad=True)
    loss = w ** 2 + 20
    print(f"Initial: w={w}, loss={loss}")

    for i in range(1, 1001):
        # Forward pass
        loss = w ** 2 + 20

        # Zero gradients (accumulate by default)
        if w.grad is not None:
            w.grad.zero_()

        # Backward pass
        loss.backward()

        # Update weight
        w.data = w.data - 0.01 * w.grad

        if i % 100 == 0:
            print(f"Step {i}: w={w.item():.4f}, grad={w.grad.item():.4f}, loss={loss.item():.4f}")

    print(f"Final: w={w.item():.4f}, loss={loss.item():.4f}")


def dm03():
    # detach()
    # once the tensor set autograde, the tensor cannot turn to ndarry
    t1 = torch.tensor([10, 20],requires_grad=True, dtype=torch.float)
    print(f't1: {t1}, type: {type(t1)}')

    n1 = t1.detach().numpy()
    print(f'n1: {n1}, type: {type(n1)}')


def dm04():
    """
    autograd use case
    1. Forward pass: compute predicted values.
    2. Loss calculation & Backward pass: compute loss between predictions and ground truth, then calculate gradients.
    3. Zero gradients: clear out old gradients from the previous iteration.
    4. Parameter update: apply the weight update formula: W_new = W_old - lr * grad.
    """
    # feature data(input)
    x = torch.ones(2, 5)
    print(f'x: {x}')

    # label data(ground truth)
    y = torch.zeros(2, 3)
    print(f'y: {y}')

    # initial weight and bias
    w = torch.randn(5, 3, requires_grad=True)
    print(f'w: {w}')
    b = torch.randn(3, requires_grad=True)
    print(f'b: {b}')

    # forward pass compute predictions
    z = torch.matmul(x, w) + b
    print(f'z: {z}')

    # define loss function
    criterion = torch.nn.MSELoss()
    loss = criterion(z, y)
    print(f'loss: {loss}')

    # autograd
    loss.backward()

    # print new w, b
    print(f'w.grad: {w.grad}')
    print(f'b.grad: {b.grad}')

    # update weight

    pass
















if __name__ == '__main__':
    # dm01()
    # print()
    # dm02()
    # dm03()
    dm04()