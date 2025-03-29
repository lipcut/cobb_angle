import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import cv2
from scipy.io import loadmat
from torchvision.datasets.vision import VisionDataset

from .landmark_utils import landmarks_rearrange


class LandmarkDataset(VisionDataset):
    def __init__(
        self,
        root: str | Path,
        train: bool = True,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(root=root, transforms=transforms)
        self.phase = "train" if train else "test"
        self.image_filenames = sorted(os.listdir(self.image_folder))
        self.data, self.targets, self.image_sizes = self._load_data()

    @property
    def image_folder(self) -> str:
        return os.path.join(self.root, "data", self.phase)

    @property
    def landmarks_folder(self) -> str:
        return os.path.join(self.root, "labels", self.phase)

    def _load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]]]:
        images = [
            cv2.imread(os.path.join(self.image_folder, image_filename))
            for image_filename in self.image_filenames
        ]

        landmarks = [
            landmarks_rearrange(
                loadmat(os.path.join(self.landmarks_folder, image_filename + ".mat"))[
                    "p2"
                ]
            )
            for image_filename in self.image_filenames
        ]

        image_sizes = [
            image.shape[:2] for image in images
        ]

        # breakpoint()

        return images, landmarks, image_sizes

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
        image = self.data[index]
        landmarks = self.targets[index]
        image_size = self.image_sizes[index]

        if self.transforms is not None:
            image, landmarks = self.transforms(image, landmarks)

        return image, landmarks, image_size

    def __len__(self):
        return len(self.data)
