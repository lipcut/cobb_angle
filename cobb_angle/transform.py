from typing import Optional, Tuple

import numpy as np
import torch
from PIL.Image import Image as PILImage
from torchvision.transforms import v2


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


def landmarks_resize(
    landmarks: torch.Tensor,
    orig_dim: Tuple[int, ...] | torch.Tensor,
    new_dim: Tuple[int, ...] | torch.Tensor,
    padding: Optional[Tuple[int, ...] | torch.Tensor] = None,
) -> torch.Tensor:
    orig_dim = torch.Tensor(orig_dim)
    new_dim = torch.Tensor(new_dim)
    if padding is None:
        padding = torch.zeros(len(orig_dim.size()))
    else:
        padding = torch.Tensor(padding)
    t_landmarks = landmarks + padding
    t_landmarks = t_landmarks * (new_dim / (orig_dim + 2 * padding))
    return t_landmarks


def landmarks_rearrange(landmarks: np.ndarray) -> np.ndarray:
    r"""rearrange the points so that the output array will be a the consective array of
    top left (tl), top right (tr), bottom left (bl), bottom right (br).
    i.e.
    output = [
        tl,
        tr,
        bl,
        br,
        ...
     ]

    where all points are 2d arrays.

    we do this by separating the points by y indexes to left and right points (2 points for each),
    then separating each pairs to top and bottom ones.
    """

    # make the array more structural so that it's easier to work with (e.g. sorting)
    # i.e.
    # [[x y], ...] -> [(x, y), ...]
    new_points = np.array(
        [(x, y) for x, y in landmarks], dtype=[("x", np.float32), ("y", np.float32)]
    )
    # sort to the lefts and the rights
    boxes = np.sort(new_points.reshape(17, 4), order="x", axis=1)
    # sort to the tops and the buttoms
    boxes = np.sort(boxes.reshape(17, 2, 2), order="y", axis=2)

    # now we have an array looks like this:
    # [[[tl bl],
    #   [tr br]],...]
    #
    # but we want this:
    # [[[tl tr],
    #   [bl br]],...]
    #
    # if there's no adjustment, then when it's flatten, the order won't be correct
    boxes = np.transpose(boxes, (0, 2, 1)).reshape(17, 4)
    # sort the arrage of points by the heights of their centers
    center_y = np.mean(boxes["y"], axis=1)
    boxes = boxes[np.argsort(center_y, axis=0)].reshape(-1)
    return np.array([[x, y] for x, y in boxes])
