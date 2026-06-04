"""
Day 01: PyTorch 张量基础
学习目标：掌握 PyTorch 张量的创建方式和基本操作
"""

import torch
import numpy as np


def dm01_tensor_creation():
    """
    练习 1: torch.tensor() 创建张量
    - 根据指定数据创建张量
    - 自动推断数据类型
    """
    print("=" * 50)
    print("练习 1: torch.tensor() 创建张量")
    print("=" * 50)

    # 从标量创建
    t1 = torch.tensor(10)
    print(f"标量: t1 = {t1}, shape = {t1.shape}, dtype = {t1.dtype}")

    # 从列表创建
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.tensor(data, dtype=float)
    print(f"列表: t2 = {t2}, shape = {t2.shape}, dtype = {t2.dtype}")

    # 从 NumPy 数组创建
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.tensor(data)
    print(f"NumPy: t3 = {t3}, shape = {t3.shape}, dtype = {t3.dtype}")


def dm02_Tensor_creation():
    """
    练习 2: torch.Tensor() 创建张量
    - 根据形状创建张量（未初始化）
    - 默认数据类型为 float32
    """
    print("\n" + "=" * 50)
    print("练习 2: torch.Tensor() 创建张量")
    print("=" * 50)

    # 从标量创建（注意：这会创建一个包含 10 个元素的张量）
    t1 = torch.Tensor(10)
    print(f"标量参数: t1 = {t1}, shape = {t1.shape}")

    # 从列表创建
    data = [[1, 2, 3], [4, 5, 6]]
    t2 = torch.Tensor(data)
    print(f"列表: t2 = {t2}, shape = {t2.shape}")

    # 从 NumPy 数组创建
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.Tensor(data)
    print(f"NumPy: t3 = {t3}, shape = {t3.shape}")

    # 指定形状创建（未初始化）
    t4 = torch.Tensor(2, 3)
    print(f"形状: t4 = {t4}, shape = {t4.shape}")


def dm03_typed_tensor():
    """
    练习 3: 指定数据类型创建张量
    - IntTensor: 整数类型
    - FloatTensor: 浮点类型
    - DoubleTensor: 双精度浮点类型
    """
    print("\n" + "=" * 50)
    print("练习 3: 指定数据类型创建张量")
    print("=" * 50)

    # IntTensor
    t1 = torch.IntTensor(10)
    print(f"IntTensor: t1 = {t1}, dtype = {t1.dtype}")

    # FloatTensor
    t2 = torch.FloatTensor([[1, 2], [3, 4]])
    print(f"FloatTensor: t2 = {t2}, dtype = {t2.dtype}")

    # 从 NumPy 创建
    data = np.random.randint(0, 10, size=(2, 3))
    t3 = torch.IntTensor(data)
    print(f"IntTensor: t3 = {t3}, dtype = {t3.dtype}")


def dm04_other_creation():
    """
    练习 4: 其他创建张量的方式
    - zeros: 全零
    - ones: 全一
    - full: 指定值
    - arange: 等差数列
    - linspace: 线性间隔
    - randn: 随机正态分布
    """
    print("\n" + "=" * 50)
    print("练习 4: 其他创建张量的方式")
    print("=" * 50)

    # 全零
    zeros = torch.zeros(3, 4)
    print(f"zeros(3,4):{zeros} : {zeros.shape}")

    # zeros_like: 根据已有张量创建同形状的全0张量
    t_ref = torch.randn(2, 3)
    zeros_like = torch.zeros_like(t_ref)
    print(f"zeros_like: {zeros_like}")

    # 全一
    ones = torch.ones(2, 3)
    print(f"ones(2,3):{ones} : {ones.shape}")

    # ones_like
    ones_like = torch.ones_like(t_ref)
    print(f"ones_like: {ones_like}")

    # 指定值
    full = torch.full((2, 3), 3.14)
    print(f"full(2,3,3.14): {full}")

    # 等差数列, 指定范围的线性张量
    arange = torch.arange(0, 10, 2)
    print(f"arange(0,10,2): {arange}")

    # 线性间隔, 等差数列
    linspace = torch.linspace(1, 10, 4)
    print(f"linspace(0,1,5): {linspace}")

    # 随机张量 均匀分布
    # torch.initial_seed() # 默认当前系统时间戳
    torch.manual_seed(3)
    t1 = torch.rand(size=(2, 3))
    print(f"t1: {t1}")

    # 随机正态分布
    randn = torch.randn(2, 3)
    print(f"randn(2,3): {randn}")

    # 创建随机整数张量
    t3 = torch.randint(low=1, high=10, size=(2, 3))
    print(f"t3: {t3}")



def dm05_tensor_attributes():
    """
    练习 5: 张量属性
    - shape: 形状
    - dtype: 数据类型
    - device: 设备
    - requires_grad: 是否需要梯度
    """
    print("\n" + "=" * 50)
    print("练习 5: 张量属性")
    print("=" * 50)

    t = torch.randn(3, 4, requires_grad=True)
    print(f"shape: {t.shape}")
    print(f"dtype: {t.dtype}")
    print(f"device: {t.device}")
    print(f"requires_grad: {t.requires_grad}")
    print(f"ndim: {t.ndim}")
    print(f"numel: {t.numel()}")  # 元素总数


def dm06_type_conversion():
    """
    练习 6: 张量类型转换
    - type(): 查看/指定类型
    - half(): float16
    - float(): float32
    - double(): float64
    - int(): int32
    - long(): int64
    - short(): int16
    """
    print("\n" + "=" * 50)
    print("练习 6: 张量类型转换")
    print("=" * 50)

    t = torch.tensor([[1.5, 2.3], [3.7, 4.9]])
    print(f"原始张量: {t}, dtype = {t.dtype}")

    # 类型转换
    print(f"half():   {t.half()}, dtype = {t.half().dtype}")
    print(f"float():  {t.float()}, dtype = {t.float().dtype}")
    print(f"double(): {t.double()}, dtype = {t.double().dtype}")

    # 整数类型
    t_int = t.int()
    print(f"int():    {t_int}, dtype = {t_int.dtype}")

    t_long = t.long()
    print(f"long():   {t_long}, dtype = {t_long.dtype}")

    t_short = t.short()
    print(f"short():  {t_short}, dtype = {t_short.dtype}")

    # type() 指定类型
    t_new = t.type(torch.FloatTensor)
    print(f"type(Float): {t_new}, dtype = {t_new.dtype}")


if __name__ == "__main__":
    dm01_tensor_creation()
    dm02_Tensor_creation()
    dm03_typed_tensor()
    dm04_other_creation()
    dm05_tensor_attributes()
    dm06_type_conversion()

    print("\n" + "=" * 50)
    print("✅ Day 01 练习完成！")
    print("=" * 50)
