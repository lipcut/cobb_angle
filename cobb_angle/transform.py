from typing import Tuple

import numpy as np
import torch
from torchvision.transforms import v2

from .landmark_utils import landmarks_rearrange, landmarks_resize


def spine_dataset16_train_transforms(
    image: np.ndarray, landmarks: np.ndarray
) -> Tuple[torch.Tensor, ...]:
    height, width, _ = image.shape
    landmarks_image = torch.zeros(height, width)
    indexes = np.transpose(landmarks).tolist()[::-1]
    indexes[0] = np.clip(indexes[0], a_min=0, a_max=height - 1).tolist()
    indexes[1] = np.clip(indexes[1], a_min=0, a_max=width - 1).tolist()
    landmarks_image[indexes] = 1.0
    basic_transforms = v2.Compose(
        [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.RandomHorizontalFlip()]
    )
    image, landmarks_image = basic_transforms(image, landmarks_image)
    image_transforms = v2.Compose(
        [
            v2.RandomPhotometricDistort(),
            v2.Resize(size=(1024, 512)),
            v2.Lambda(lambd=lambda x: torch.clamp(x, min=0, max=255) / 255),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = image_transforms(image)
    landmarks = torch.argwhere(landmarks_image.squeeze() == 1)
    landmarks = torch.flip(landmarks, dims=(-1,))
    landmarks = landmarks_resize(
        landmarks, orig_dim=(height, width), new_dim=(512, 256)
    )
    landmarks = torch.tensor(landmarks_rearrange(landmarks)).to(device=landmarks.device)

    return image, landmarks


def spine_dataset16_test_transforms(
    image: np.ndarray, landmarks: np.ndarray
) -> Tuple[torch.Tensor, ...]:
    height, width, _ = image.shape
    landmarks = torch.tensor(landmarks)
    transforms = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Resize(size=(1024, 512)),
            v2.Lambda(lambd=lambda x: torch.clamp(x, min=0, max=255) / 255),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    image = transforms(image)
    landmarks = landmarks_resize(
        landmarks, orig_dim=(height, width), new_dim=(512, 256)
    )
    landmarks = torch.tensor(landmarks_rearrange(landmarks)).to(device=landmarks.device)

    return image, landmarks
