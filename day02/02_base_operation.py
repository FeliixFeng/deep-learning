"""
    Basic Tensor Operations:
    add()  sub()  mul()  div()  neg()  pow()  matmul()

    Aggregation:
    sum()  mean()  max()  min()  argmax()  argmin()

    dim parameter:
    Specifies which dimension to operate on.
    - dim=0: operate along rows (collapse rows)
    - dim=1: operate along columns (collapse columns)
    - dim=None (default): operate on all elements
"""

import torch


def dm01_add():
    """
    Addition:
    - torch.add(a, b) or a.add(b) -> new tensor
    - a.add_(b) -> inplace (modifies a)
    - a + b -> operator overload
    """
    t1 = torch.tensor([1, 2, 3])
    t2 = torch.tensor([10, 20, 30])

    # Method 1: torch.add()
    result1 = torch.add(t1, t2)
    print(f"torch.add: {result1}")

    # Method 2: tensor.add()
    result2 = t1.add(t2)
    print(f"tensor.add: {result2}")

    # Method 3: operator
    result3 = t1 + t2
    print(f"operator +: {result3}")

    # Inplace: modifies t1 directly
    print(f"\nBefore inplace: t1 = {t1}")
    t1.add_(t2)
    print(f"After inplace:  t1 = {t1}")


def dm02_sub():
    """
    Subtraction:
    - torch.sub() / tensor.sub() / operator -
    - Inplace: sub_()
    """
    t1 = torch.tensor([100, 200, 300])
    t2 = torch.tensor([1, 2, 3])

    result = t1 - t2
    print(f"subtraction: {result}")

    # Inplace
    t1.sub_(t2)
    print(f"inplace sub: {t1}")


def dm03_mul():
    """
    Multiplication (element-wise):
    - torch.mul() / tensor.mul() / operator *
    - NOT matrix multiply! Use torch.matmul() or @ for that.
    """
    t1 = torch.tensor([2, 3, 4])
    t2 = torch.tensor([5, 6, 7])

    result = t1 * t2
    print(f"element-wise mul: {result}")
    print(f"torch.mul: {torch.mul(t1, t2)}")


def dm04_div():
    """
    Division (element-wise):
    - torch.div() / tensor.div() / operator /
    """
    t1 = torch.tensor([10.0, 20.0, 30.0])
    t2 = torch.tensor([2.0, 4.0, 5.0])

    result = t1 / t2
    print(f"division: {result}")


def dm05_neg():
    """
    Negation:
    - torch.neg() / tensor.neg()
    """
    t1 = torch.tensor([1, -2, 3, -4])

    result = torch.neg(t1)
    print(f"neg: {result}")


def dm06_power():
    """
    Power:
    - torch.pow() / tensor.pow() / operator **
    """
    t1 = torch.tensor([2, 3, 4])

    result = t1 ** 2
    print(f"power: {result}")


def dm07_matmul():
    """
    Matrix multiplication:
    - torch.matmul(a, b)
    - a @ b
    """
    t1 = torch.tensor([[1, 2], [3, 4]])
    t2 = torch.tensor([[5, 6], [7, 8]])

    result1 = torch.matmul(t1, t2)
    result2 = t1 @ t2
    print(f"matmul:\n{result1}")
    print(f"@ operator:\n{result2}")
    print(f"equal: {torch.equal(result1, result2)}")


def dm08_aggregation():
    """
    Aggregation functions:
    - sum(): sum of all elements
    - mean(): average of all elements
    - max(): maximum element
    - min(): minimum element
    - argmax(): index of maximum element
    - argmin(): index of minimum element
    """
    print("=" * 50)
    print("Aggregation Functions")
    print("=" * 50)

    t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print(f"tensor:\n{t}")

    # Basic aggregation
    print(f"\nsum: {t.sum()}")
    print(f"mean: {t.mean()}")
    print(f"max: {t.max()}")
    print(f"min: {t.min()}")
    print(f"argmax: {t.argmax()}")  # flattened index
    print(f"argmin: {t.argmin()}")  # flattened index


def dm08_5_cat():
    """
    cat(): concatenate tensors along an existing dimension
    - torch.cat([t1, t2], dim=0) -> stack vertically (add rows)
    - torch.cat([t1, t2], dim=1) -> stack horizontally (add columns)
    """
    print("\n" + "=" * 50)
    print("cat() - Concatenate")
    print("=" * 50)

    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    t2 = torch.tensor([[7, 8, 9], [10, 11, 12]])
    print(f"t1:\n{t1}")
    print(f"t2:\n{t2}")

    # dim=0: vertical concat (add rows)
    print(f"\ncat dim=0:\n{torch.cat([t1, t2], dim=0)}")

    # dim=1: horizontal concat (add columns)
    print(f"\ncat dim=1:\n{torch.cat([t1, t2], dim=1)}")


def dm09_dim_parameter():
    """
    dim parameter explained:

    For a 2D tensor with shape (2, 3):
    - dim=0: along rows -> collapse rows -> result shape (3,)
    - dim=1: along columns -> collapse columns -> result shape (2,)

    Think of it as: which axis do you want to "squeeze out"?
    """
    print("\n" + "=" * 50)
    print("dim Parameter Explained")
    print("=" * 50)

    t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    print(f"tensor (shape {t.shape}):\n{t}")

    # dim=0: operate along rows (collapse dimension 0)
    # Result: sum each column -> [1+4, 2+5, 3+6] = [5, 7, 9]
    print(f"\nsum(dim=0): {t.sum(dim=0)}")  # [5, 7, 9]
    print(f"mean(dim=0): {t.mean(dim=0)}")  # [2.5, 3.5, 4.5]
    print(f"max(dim=0): {t.max(dim=0)}")  # values: [4, 5, 6]

    # dim=1: operate along columns (collapse dimension 1)
    # Result: sum each row -> [1+2+3, 4+5+6] = [6, 15]
    print(f"\nsum(dim=1): {t.sum(dim=1)}")  # [6, 15]
    print(f"mean(dim=1): {t.mean(dim=1)}")  # [2.0, 5.0]
    print(f"max(dim=1): {t.max(dim=1)}")  # values: [3, 6]

    # keepdim=True: keeps the reduced dimension as size 1
    print(f"\nsum(dim=0, keepdim=True): {t.sum(dim=0, keepdim=True)}")
    print(f"sum(dim=1, keepdim=True): {t.sum(dim=1, keepdim=True)}")


def dm10_dim_3d():
    """
    dim in 3D tensors (common in deep learning):
    - dim=0: batch dimension
    - dim=1: sequence/height dimension
    - dim=2: feature/width dimension
    """
    print("\n" + "=" * 50)
    print("dim in 3D Tensor")
    print("=" * 50)

    # Shape: (2, 3, 4) -> 2 batches, 3 rows, 4 columns
    t = torch.arange(24).reshape(2, 3, 4).float()
    print(f"tensor (shape {t.shape}):\n{t}")

    print(f"\nsum(dim=0): shape {t.sum(dim=0).shape}")  # (3, 4)
    print(f"sum(dim=1): shape {t.sum(dim=1).shape}")  # (2, 4)
    print(f"sum(dim=2): shape {t.sum(dim=2).shape}")  # (2, 3)


if __name__ == '__main__':
    dm01_add()
    print()
    dm02_sub()
    print()
    dm03_mul()
    print()
    dm04_div()
    print()
    dm05_neg()
    print()
    dm06_power()
    print()
    dm07_matmul()
    print()
    dm08_aggregation()
    dm08_5_cat()
    dm09_dim_parameter()
    dm10_dim_3d()