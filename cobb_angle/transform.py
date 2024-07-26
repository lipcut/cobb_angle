from typing import Tuple

import numpy as np
import torch
from PIL.Image import Image as PILImage
from torchvision.transforms import v2

from .utils import landmarks_rearrange, landmarks_resize


def spinet16_train_transforms(
    image: PILImage, landmarks: np.ndarray
) -> Tuple[torch.Tensor, ...]:
    width, height = image.size
    landmarks_image = torch.zeros(height, width)
    indexes = np.transpose(landmarks).tolist()
    landmarks_image[indexes] = 1.0
    basic_transforms = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.RandomHorizontalFlip()]
    )
    image, landmarks_image = basic_transforms(image, landmarks_image)
    image = torch.clamp(image, min=0.0, max=255.0) / 255.0
    image_transforms = v2.Compose(
        [
            v2.RandomPhotometricDistort(),
            v2.Resize(size=(1024, 512)),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = image_transforms(image)
    landmarks = torch.argwhere(landmarks_image.squeeze() == 1)
    landmarks = landmarks_resize(
        landmarks, orig_dim=(height, width), new_dim=(1024, 512)
    )

    return image, landmarks


def spinet16_val_transforms(
    image: PILImage, landmarks: np.ndarray
) -> Tuple[torch.Tensor, ...]:
    width, height = image.size
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(1024, 512)),
        ]
    )
    image = transforms(image)
    image = torch.clamp(image, min=0.0, max=255.0) / 255.0
    landmarks = landmarks_resize(
        landmarks, orig_dim=(height, width), new_dim=(1024, 512)
    )
    landmarks = landmarks_rearrange(landmarks)

    return image, landmarks


def spinenet16_test_transforms(image: PILImage) -> torch.Tensor:
    transforms = v2.Compose(
        [
            v2.PILToTensor(),
            v2.Resize(size=(1024, 512)),
            v2.Lambda(lambd=lambda x: x / 255),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    return transforms(image)
