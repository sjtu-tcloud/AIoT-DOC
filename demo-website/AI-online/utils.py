import numpy as np
import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


def imshow(sample_batch):
    inputs, labels = sample_batch
    images_transformed = make_grid(inputs, nrow=4, pad_value=255)
    images_transformed = np.transpose(images_transformed.numpy(), (1, 2, 0))
    plt.imshow(images_transformed)
