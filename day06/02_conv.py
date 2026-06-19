"""
    Convolutional Neural Network (CNN)
    - Convolutional layer
    - Pooling layer
    - Fully Connected layer

    Feature map size formula:
    N = (W - F + 2*P) / S + 1
    W: input size, F: filter size, P: padding, S: stride
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def dm01():
    """
    Basic convolution operation:
    1. Load image (HWC format)
    2. Convert to tensor and permute to CHW
    3. Add batch dimension (NCHW)
    4. Apply convolution
    5. Visualize feature map
    """
    # 1. Load image (HWC: Height, Width, Channels)
    img = plt.imread('./data/img.jpg')
    print(f'Original image shape: {img.shape}')  # (640, 640, 3)

    # 2. Convert to tensor and permute HWC -> CHW
    img2 = torch.tensor(img, dtype=torch.float)
    img2 = img2.permute(2, 0, 1)  # (3, 640, 640)

    # 3. Add batch dimension: CHW -> NCHW (1, 3, 640, 640)
    img3 = img2.unsqueeze(dim=0)

    # 4. Create convolution layer
    # in_channels=3 (RGB), out_channels=4 (filters), kernel_size=3, stride=2, padding=0
    conv = nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, stride=2, padding=0)

    # 5. Apply convolution
    conv_img = conv(img3)
    print(f'After conv shape: {conv_img.shape}')  # (1, 4, 319, 319)

    # 6. Extract and visualize feature map
    # Remove batch dimension: (1, 4, 319, 319) -> (4, 319, 319)
    img4 = conv_img[0]

    # Permute for display: CHW -> HWC
    img5 = img4.permute(1, 2, 0)

    # Show feature map (channel 3)
    feature1 = img5[:, :, 3].detach().numpy()
    plt.imshow(feature1, cmap='gray')
    plt.title('Feature Map (Channel 3)')
    plt.show()






if __name__ == '__main__':
    dm01()