"""
    Shape Operations:
    1. reshape() / view()  -> change shape
    2. squeeze()           -> remove size-1 dimensions
    3. unsqueeze()         -> add size-1 dimensions
    4. transpose()         -> swap two dimensions
    5. permute()           -> reorder dimensions
    6. contiguous()        -> make memory contiguous
"""

import torch


def dm01_reshape_view():
    """
    reshape() and view() both change tensor shape.
    - view() requires contiguous memory
    - reshape() works on non-contiguous (may copy)
    - -1 means "infer this dimension"
    """
    print("=" * 50)
    print("1. reshape() / view()")
    print("=" * 50)

    t = torch.arange(12)
    print(f"original (shape {t.shape}): {t}")

    # reshape
    t2 = t.reshape(3, 4)
    print(f"\nreshape(3,4):\n{t2}")

    t3 = t.reshape(4, 3)
    print(f"reshape(4,3):\n{t3}")

    # -1 means auto-calculate
    t4 = t.reshape(2, -1)  # 12 / 2 = 6
    print(f"reshape(2,-1):\n{t4}")

    t5 = t.reshape(-1, 4)  # 12 / 4 = 3
    print(f"reshape(-1,4):\n{t5}")

    # view() is similar
    t6 = t.view(3, 4)
    print(f"\nview(3,4):\n{t6}")

    # flatten shortcut
    print(f"\nflatten: {t.reshape(-1)}")


def dm06_stack():
    """
    stack(): join tensors along a NEW dimension
    - Unlike cat(), stack adds a new dimension
    - cat() concatenates along existing dimension
    """
    print("\n" + "=" * 50)
    print("6. stack()")
    print("=" * 50)

    t1 = torch.tensor([1, 2, 3])
    t2 = torch.tensor([4, 5, 6])
    print(f"t1: {t1}")
    print(f"t2: {t2}")

    # stack along new dim 0 -> shape (2, 3)
    t3 = torch.stack([t1, t2])
    print(f"\nstack dim=0 (shape {t3.shape}):\n{t3}")

    # stack along new dim 1 -> shape (3, 2)
    t4 = torch.stack([t1, t2], dim=1)
    print(f"\nstack dim=1 (shape {t4.shape}):\n{t4}")

    # Compare with cat (no new dimension)
    t5 = torch.cat([t1.unsqueeze(0), t2.unsqueeze(0)])
    print(f"\ncat (shape {t5.shape}):\n{t5}")


def dm02_squeeze_unsqueeze():
    """
    squeeze(): remove dimensions of size 1
    unsqueeze(): add a dimension of size 1
    """
    print("\n" + "=" * 50)
    print("2. squeeze() / unsqueeze()")
    print("=" * 50)

    # squeeze
    t = torch.tensor([[[1, 2, 3]]])  # shape (1, 1, 3)
    print(f"original (shape {t.shape}): {t}")
    print(f"squeeze() (shape {t.squeeze().shape}): {t.squeeze()}")

    # unsqueeze
    t2 = torch.tensor([1, 2, 3])  # shape (3,)
    print(f"\noriginal (shape {t2.shape}): {t2}")
    print(f"unsqueeze(0) (shape {t2.unsqueeze(0).shape}):\n{t2.unsqueeze(0)}")
    print(f"unsqueeze(1) (shape {t2.unsqueeze(1).shape}):\n{t2.unsqueeze(1)}")


def dm03_transpose():
    """
    transpose(dim0, dim1): swap two dimensions
    For 2D: t.t() is shorthand for transpose(0, 1)
    """
    print("\n" + "=" * 50)
    print("3. transpose()")
    print("=" * 50)

    t = torch.tensor([[1, 2, 3],
                      [4, 5, 6]])
    print(f"original (shape {t.shape}):\n{t}")

    t2 = t.transpose(0, 1)
    print(f"\ntranspose(0,1) (shape {t2.shape}):\n{t2}")

    # shorthand for 2D
    print(f"\nt.t() (shape {t.t().shape}):\n{t.t()}")


def dm04_permute():
    """
    permute(dims): reorder dimensions arbitrarily
    Common in image processing: (C, H, W) -> (H, W, C)
    """
    print("\n" + "=" * 50)
    print("4. permute()")
    print("=" * 50)

    # Example: image tensor (batch, channels, height, width)
    t = torch.randn(2, 3, 4, 5)  # (B, C, H, W)
    print(f"original (shape {t.shape}): (B, C, H, W)")

    # Reorder to (B, H, W, C)
    t2 = t.permute(0, 2, 3, 1)
    print(f"permute(0,2,3,1) (shape {t2.shape}): (B, H, W, C)")

    # Simple 2D example
    t3 = torch.tensor([[1, 2, 3], [4, 5, 6]])
    print(f"\noriginal (shape {t3.shape}):\n{t3}")
    print(f"permute(1,0) (shape {t3.permute(1,0).shape}):\n{t3.permute(1,0)}")


def dm05_contiguous():
    """
    contiguous(): makes memory layout contiguous
    After transpose/permute, tensor may be non-contiguous
    view() requires contiguous memory
    """
    print("\n" + "=" * 50)
    print("5. contiguous()")
    print("=" * 50)

    t = torch.arange(6).reshape(2, 3)
    print(f"original (shape {t.shape}):\n{t}")

    # transpose makes it non-contiguous
    t2 = t.transpose(0, 1)
    print(f"\ntranspose (contiguous? {t2.is_contiguous()}):\n{t2}")

    # view() fails on non-contiguous
    try:
        t2.view(6)
    except Exception as e:
        print(f"\nview() error: {e}")

    # fix with contiguous()
    t3 = t2.contiguous().view(6)
    print(f"contiguous().view(6): {t3}")


def dm07_practice():
    """
    Quick practice: common shape operations in deep learning
    """
    print("\n" + "=" * 50)
    print("7. Common Patterns")
    print("=" * 50)

    # Add batch dimension
    t = torch.randn(3, 4)  # (3, 4)
    print(f"original (shape {t.shape})")
    t_batch = t.unsqueeze(0)  # (1, 3, 4)
    print(f"add batch (shape {t_batch.shape})")

    # Remove batch dimension
    t_back = t_batch.squeeze(0)
    print(f"remove batch (shape {t_back.shape})")

    # Reshape for linear layer
    t_img = torch.randn(1, 3, 28, 28)  # (B, C, H, W)
    t_flat = t_img.reshape(1, -1)  # (1, 2352)
    print(f"\nimage (shape {t_img.shape})")
    print(f"flatten for linear (shape {t_flat.shape})")


if __name__ == '__main__':
    dm01_reshape_view()
    dm02_squeeze_unsqueeze()
    dm03_transpose()
    dm04_permute()
    dm05_contiguous()
    dm06_stack()
    dm07_practice()
