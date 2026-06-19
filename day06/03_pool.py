"""
    Pooling Layers
    - MaxPooling: 取最大值
    - AvgPooling: 取平均值

    Pooling作用：
    1. 降低特征图尺寸（降维）
    2. 减少参数量
    3. 提取主要特征
    4. 防止过拟合
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def dm01():
    """
    MaxPooling: 取窗口内的最大值
    """
    print("=" * 50)
    print("1. MaxPooling")
    print("=" * 50)

    # Create input tensor (1, 1, 4, 4)
    x = torch.tensor([[[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]]]], dtype=torch.float32)

    print(f"Input shape: {x.shape}")
    print(f"Input:\n{x.squeeze()}")

    # MaxPooling with kernel_size=2, stride=2
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    output = max_pool(x)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output:\n{output.squeeze()}")


def dm02():
    """
    AvgPooling: 取窗口内的平均值
    """
    print("\n" + "=" * 50)
    print("2. AvgPooling")
    print("=" * 50)

    # Create input tensor (1, 1, 4, 4)
    x = torch.tensor([[[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]]]], dtype=torch.float32)

    print(f"Input shape: {x.shape}")
    print(f"Input:\n{x.squeeze()}")

    # AvgPooling with kernel_size=2, stride=2
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    output = avg_pool(x)

    print(f"\nOutput shape: {output.shape}")
    print(f"Output:\n{output.squeeze()}")


def dm03():
    """
    Global Average Pooling: 对整个特征图取平均
    常用于分类网络的最后一层
    """
    print("\n" + "=" * 50)
    print("3. Global Average Pooling")
    print("=" * 50)

    # Create input tensor (1, 3, 4, 4)
    x = torch.randn(1, 3, 4, 4)
    print(f"Input shape: {x.shape}")

    # Global Average Pooling
    global_pool = nn.AdaptiveAvgPool2d(1)
    output = global_pool(x)

    print(f"Output shape: {output.shape}")  # (1, 3, 1, 1)
    print(f"Output flattened: {output.flatten().shape}")  # (3,)


def dm04():
    """
    Compare MaxPooling vs AvgPooling
    """
    print("\n" + "=" * 50)
    print("4. Compare MaxPooling vs AvgPooling")
    print("=" * 50)

    # Create input tensor
    x = torch.tensor([[[[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12],
                        [13, 14, 15, 16]]]], dtype=torch.float32)

    # MaxPooling
    max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    max_output = max_pool(x)

    # AvgPooling
    avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
    avg_output = avg_pool(x)

    print(f"Input:\n{x.squeeze()}")
    print(f"\nMaxPooling output:\n{max_output.squeeze()}")
    print(f"\nAvgPooling output:\n{avg_output.squeeze()}")
    print(f"\nMax > Avg: {(max_output > avg_output).all().item()}")


if __name__ == '__main__':
    dm01()
    dm02()
    dm03()
    dm04()
    