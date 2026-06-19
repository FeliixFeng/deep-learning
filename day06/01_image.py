"""
    Image Processing Basics
    案例：演示基础的图像操作

    图像分类 (Image Types):
    1. 二值图 (Binary): 1通道, 每个像素点由0, 1组成
    2. 灰度图 (Grayscale): 1通道, 每个像素点的范围: [0, 255]
    3. 索引图 (Indexed): 1通道, 每个像素点的范围: [0, 255], 像素点表示颜色表的索引
    4. RGB真彩图 (RGB): 3通道 (R, G, B), 每个通道范围: [0, 255]
"""

import numpy as np
import matplotlib.pyplot as plt


def dm01_binary_image():
    """
    1. 二值图 (Binary Image)
    - 1 channel
    - Each pixel is 0 or 1
    """
    print("=" * 50)
    print("1. Binary Image")
    print("=" * 50)

    # Create binary image (8x8)
    binary = np.array([
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1],
        [1, 1, 0, 0, 1, 1, 0, 0],
        [1, 1, 0, 0, 1, 1, 0, 0]
    ], dtype=np.uint8)

    print(f"Shape: {binary.shape}")
    print(f"Unique values: {np.unique(binary)}")
    print(f"Data type: {binary.dtype}")

    # Display
    plt.figure()
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image')
    plt.colorbar()
    plt.show()


def dm02_grayscale_image():
    """
    2. 灰度图 (Grayscale Image)
    - 1 channel
    - Pixel range: [0, 255]
    """
    print("\n" + "=" * 50)
    print("2. Grayscale Image")
    print("=" * 50)

    # Create grayscale image (8x8)
    grayscale = np.random.randint(0, 256, size=(8, 8), dtype=np.uint8)

    print(f"Shape: {grayscale.shape}")
    print(f"Min pixel: {grayscale.min()}, Max pixel: {grayscale.max()}")
    print(f"Data type: {grayscale.dtype}")

    # Display
    plt.figure()
    plt.imshow(grayscale, cmap='gray')
    plt.title('Grayscale Image')
    plt.colorbar()
    plt.show()


def dm03_rgb_image():
    """
    3. RGB真彩图 (RGB Color Image)
    - 3 channels (R, G, B)
    - Each channel range: [0, 255]
    """
    print("\n" + "=" * 50)
    print("3. RGB Image")
    print("=" * 50)

    # Create RGB image (8x8x3)
    rgb = np.random.randint(0, 256, size=(8, 8, 3), dtype=np.uint8)

    print(f"Shape: {rgb.shape}")
    print(f"R channel range: [{rgb[:,:,0].min()}, {rgb[:,:,0].max()}]")
    print(f"G channel range: [{rgb[:,:,1].min()}, {rgb[:,:,1].max()}]")
    print(f"B channel range: [{rgb[:,:,2].min()}, {rgb[:,:,2].max()}]")

    # Display
    plt.figure()
    plt.imshow(rgb)
    plt.title('RGB Image')
    plt.show()


def dm04_image_channels():
    """
    4. Split and merge RGB channels
    """
    print("\n" + "=" * 50)
    print("4. Image Channels")
    print("=" * 50)

    # Create RGB image
    rgb = np.random.randint(0, 256, size=(8, 8, 3), dtype=np.uint8)

    # Split channels
    r_channel = rgb[:, :, 0]
    g_channel = rgb[:, :, 1]
    b_channel = rgb[:, :, 2]

    print(f"R channel shape: {r_channel.shape}")
    print(f"G channel shape: {g_channel.shape}")
    print(f"B channel shape: {b_channel.shape}")

    # Display channels
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].imshow(rgb)
    axes[0].set_title('RGB')
    axes[1].imshow(r_channel, cmap='Reds')
    axes[1].set_title('Red')
    axes[2].imshow(g_channel, cmap='Greens')
    axes[2].set_title('Green')
    axes[3].imshow(b_channel, cmap='Blues')
    axes[3].set_title('Blue')
    plt.tight_layout()
    plt.show()


def dm05_practice():
    """
    5. Practice: Create a simple pattern image
    """
    print("\n" + "=" * 50)
    print("5. Practice: Create Pattern")
    print("=" * 50)

    # Create a gradient image
    gradient = np.zeros((100, 100), dtype=np.uint8)
    for i in range(100):
        gradient[i, :] = i * 255 // 100

    print(f"Shape: {gradient.shape}")
    print(f"Min: {gradient.min()}, Max: {gradient.max()}")

    # Display
    plt.figure()
    plt.imshow(gradient, cmap='gray')
    plt.title('Gradient Image')
    plt.colorbar()
    plt.show()


def dm():
    """
    Read and display image from file
    - plt.imread() returns numpy array (HWC format)
    - Shape: (height, width, channels) for RGB
    """
    # Read image (returns HWC format: Height, Width, Channels)
    img1 = plt.imread('./data/img.jpg')
    print(f'Image shape: {img1.shape}')  # (H, W, C)
    print(f'Data type: {img1.dtype}')
    print(f'Pixel range: [{img1.min()}, {img1.max()}]')

    # Display image
    plt.imshow(img1)
    plt.title('Original Image')
    plt.show()


if __name__ == '__main__':
    # dm01_binary_image()
    # dm02_grayscale_image()
    # dm03_rgb_image()
    # dm04_image_channels()
    # dm05_practice()
    dm()