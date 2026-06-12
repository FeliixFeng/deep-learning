"""
    Weight Initialization:
    1. Zero / Constant init
    2. Random init (uniform, normal)
    3. Xavier/Glorot init
    4. Kaiming/He init
    5. When to use which
"""

import torch
import torch.nn as nn


def dm01_zero_init():
    """
    Zero initialization: all weights = 0
    Problem: all neurons learn the same thing (symmetry breaking issue)
    """
    print("=" * 50)
    print("1. Zero Initialization")
    print("=" * 50)

    linear = nn.Linear(5, 3)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    print(f"weight:\n{linear.weight.data}")
    print(f"bias: {linear.bias.data}")


def dm02_random_init():
    """
    Random initialization: break symmetry
    - Uniform: uniform(low, high)
    - Normal: normal(mean, std)
    """
    print("\n" + "=" * 50)
    print("2. Random Initialization")
    print("=" * 50)

    linear = nn.Linear(5, 3)

    # Uniform distribution [-0.5, 0.5]
    nn.init.uniform_(linear.weight, -0.5, 0.5)
    print(f"uniform: {linear.weight.data}")

    # Normal distribution mean=0, std=1
    nn.init.normal_(linear.weight, mean=0, std=1)
    print(f"normal: {linear.weight.data}")


def dm03_xavier_init():
    """
    Xavier/Glorot initialization:
    - Keeps variance stable across layers
    - Good for sigmoid/tanh activation
    - weight ~ Uniform(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
    """
    print("\n" + "=" * 50)
    print("3. Xavier Initialization")
    print("=" * 50)

    linear = nn.Linear(256, 128)

    # Xavier uniform
    nn.init.xavier_uniform_(linear.weight)
    print(f"xavier_uniform: mean={linear.weight.mean():.4f}, std={linear.weight.std():.4f}")

    # Xavier normal
    nn.init.xavier_normal_(linear.weight)
    print(f"xavier_normal: mean={linear.weight.mean():.4f}, std={linear.weight.std():.4f}")


def dm04_kaiming_init():
    """
    Kaiming/He initialization:
    - Designed for ReLU activation
    - Keeps variance stable through ReLU
    - weight ~ Normal(0, sqrt(2/fan_in))
    """
    print("\n" + "=" * 50)
    print("4. Kaiming Initialization")
    print("=" * 50)

    linear = nn.Linear(256, 128)

    # Kaiming uniform (for ReLU)
    nn.init.kaiming_uniform_(linear.weight, nonlinearity='relu')
    print(f"kaiming_uniform: mean={linear.weight.mean():.4f}, std={linear.weight.std():.4f}")

    # Kaiming normal (for ReLU)
    nn.init.kaiming_normal_(linear.weight, nonlinearity='relu')
    print(f"kaiming_normal: mean={linear.weight.mean():.4f}, std={linear.weight.std():.4f}")


def dm05_custom_init():
    """
    Custom initialization example:
    Apply different init to different layers
    """
    print("\n" + "=" * 50)
    print("5. Custom Initialization")
    print("=" * 50)

    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)

    model = nn.Sequential(
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10)
    )

    model.apply(init_weights)

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name}: shape={param.shape}, mean={param.mean():.4f}")


if __name__ == '__main__':
    dm01_zero_init()
    dm02_random_init()
    dm03_xavier_init()
    dm04_kaiming_init()
    dm05_custom_init()
