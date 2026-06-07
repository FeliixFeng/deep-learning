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


if __name__ == '__main__':
    dm01()
    print()
    dm02()