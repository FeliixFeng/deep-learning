import numpy as np
import torch

"""
    1. Tensor -> NumPy
    tensor.numpy()       share memory
    tensor.numpy().copy() no share memory
"""

def dm01():
    t1 = torch.tensor([1, 2, 3, 4, 5])
    print(f't1:{t1}, type: {type(t1)}')

    # Share memory: n1 and t1 point to the same data
    # n1 = t1.numpy()

    # No share: n1 is a copy, independent from t1
    n1 = t1.numpy().copy()
    print(f'n1:{n1}, type: {type(n1)}')

    # Modify n1, t1 unchanged (because we used .copy())
    n1[0] = 100
    print(f'n1: {n1}')
    print(f't1: {t1}')


"""
    2. NumPy -> Tensor
    torch.from_numpy()       share memory
    torch.tensor(ndarray)    no share memory
"""

def dm02():
    n1 = np.array([11, 22, 33])
    print(f'n1:{n1}, type: {type(n1)}')

    # Share memory (uncomment to test):
    # t1 = torch.from_numpy(n1)

    # No share memory
    t1 = torch.tensor(n1)
    print(f't1: {t1}, type: {type(t1)}')

    # Modify n1, t1 unchanged (because we used torch.tensor())
    n1[0] = 100
    print(f'n1: {n1}')
    print(f't1: {t1}')


"""
    3. Extract item from scalar tensor
    tensor.item() -> Python scalar
"""

def dm03():
    t1 = torch.tensor(100)
    value = t1.item()
    print(f'value: {value}, type: {type(value)}')


if __name__ == '__main__':
    # dm01()
    # dm02()
    dm03()
