"""
    Tensor Indexing & Slicing:

    1. 简单行列索引: t[0, 1]
    2. 列表索引: t[[0,1], [1,2]] -> select specific positions
    3. 范围索引: t[:3, :2] -> slice ranges
    4. 布尔索引: t[t > 5] -> conditional selection
    5. 多维索引: 3D+ tensor indexing by axis
"""

import torch

torch.manual_seed(24)


def dm01_basic_indexing():
    """
    1. 简单行列索引
    - t[i] -> select row i
    - t[i, j] -> select element at (i, j)
    """
    print("=" * 50)
    print("1. 简单行列索引")
    print("=" * 50)

    t = torch.randint(1, 10, (5, 5))
    print(f"tensor:\n{t}")

    # Row indexing
    print(f"\nt[0]: {t[0]}")
    print(f"t[1]: {t[1]}")

    # Element indexing
    print(f"t[0, 1]: {t[0, 1]}")

    # Colon : means "all"
    print(f"\nt[:] (all): {t[:]}")
    print(f"t[:, 0] (all rows, col 0): {t[:, 0]}")
    print(f"t[0, :] (row 0, all cols): {t[0, :]}")

    # Negative indexing: -1 means last
    print(f"\nt[-1] (last row): {t[-1]}")
    print(f"t[:, -1] (last column): {t[:, -1]}")
    print(f"t[-1, -1] (last element): {t[-1, -1]}")
    print(f"t[:-1] (all except last row):\n{t[:-1]}")
    print(f"t[2, 2]: {t[2, 2]}")  # 90


def dm02_list_indexing():
    """
    2. 列表索引
    - t[[row_indices], [col_indices]] -> select specific (row, col) pairs
    - Returns elements at those positions
    """
    print("\n" + "=" * 50)
    print("2. 列表索引")
    print("=" * 50)

    t = torch.tensor([[10, 20, 30],
                      [40, 50, 60],
                      [70, 80, 90]])
    print(f"tensor:\n{t}")

    # Select specific positions: (0,1) and (1,2)
    print(f"\nt[[0, 1], [1, 2]]: {t[[0, 1], [1, 2]]}")  # [20, 60]

    # Select rows 0,1 and columns 1,2 -> 4 elements
    print(f"\nt[[0,1]][:, [1,2]]:\n{t[[0,1]][:, [1,2]]}")


def dm03_range_indexing():
    """
    3. 范围索引
    - t[:3, :2] -> first 3 rows, first 2 columns
    - t[2:, :2] -> from row 2, first 2 columns
    """
    print("\n" + "=" * 50)
    print("3. 范围索引")
    print("=" * 50)

    t = torch.tensor([[10, 20, 30],
                      [40, 50, 60],
                      [70, 80, 90]])
    print(f"tensor:\n{t}")

    # First 3 rows, first 2 columns
    print(f"\nt[:3, :2] (前3行的前2列):\n{t[:3, :2]}")

    # From row 2, first 2 columns
    print(f"\nt[2:, :2] (第2行到最后的前2列):\n{t[2:, :2]}")

    # First 2 rows, last 2 columns
    print(f"\nt[:2, -2:] (前2行的后2列):\n{t[:2, -2:]}")

    # Step slicing
    print(f"\nt[::2, ::2] (每隔1行/列):\n{t[::2, ::2]}")


def dm04_boolean_indexing():
    """
    4. 布尔索引
    - t[t > 5] -> elements where condition is True
    - t[t[:, col] > 5] -> rows where column value > 5
    """
    print("\n" + "=" * 50)
    print("4. 布尔索引")
    print("=" * 50)

    t = torch.tensor([[10, 20, 30],
                      [40, 50, 60],
                      [70, 80, 90]])
    print(f"tensor:\n{t}")

    # All elements > 50
    print(f"\nt[t > 50] (所有大于5的元素): {t[t > 50]}")

    # Rows where column 2 > 50
    print(f"\nt[t[:, 2] > 50] (第3列大于50的行):\n{t[t[:, 2] > 50]}")

    # Columns where row 1 > 40
    print(f"\nt[:, t[1, :] > 40] (第2行大于40的列):\n{t[:, t[1, :] > 40]}")


def dm05_multi_dim_indexing():
    """
    5. 多维索引 (3D+)
    - axis 0: batch dimension
    - axis 1: row/height dimension
    - axis 2: column/width dimension
    """
    print("\n" + "=" * 50)
    print("5. 多维索引")
    print("=" * 50)

    t = torch.arange(24).reshape(2, 3, 4).float()
    print(f"tensor (shape {t.shape}):\n{t}")

    # Get first element on each axis
    print(f"\nt[0] (axis 0, 第一个batch):\n{t[0]}")
    print(f"\nt[:, 0] (axis 1, 第一个row):\n{t[:, 0]}")
    print(f"\nt[:, :, 0] (axis 2, 第一个col):\n{t[:, :, 0]}")

    # Single element in 3D
    print(f"\nt[0, 1, 2] (batch=0, row=1, col=2): {t[0, 1, 2]}")


if __name__ == '__main__':
    dm01_basic_indexing()
    dm02_list_indexing()
    dm03_range_indexing()
    dm04_boolean_indexing()
    dm05_multi_dim_indexing()
