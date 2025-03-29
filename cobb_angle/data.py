import os
from pathlib import Path
from typing import Callable, Optional, Tuple

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
        self.data, self.targets = self._load_data()

    @property
    def image_folder(self) -> Path:
        return Path(self.root, "data", self.phase)

    @property
    def landmarks_folder(self) -> Path:
        return Path(self.root, "labels", self.phase)

    def _load_data(self) -> Tuple[list[np.ndarray], list[np.ndarray]]:
        images, landmarks = [], []

        for image_filename in self.image_filenames:
            cv2_image = cv2.imread(Path(self.image_folder, image_filename))
            images.append(cv2_image)

        for image_filename in self.image_filenames:
            landmark = loadmat(Path(self.landmarks_folder, image_filename + ".mat"))["p2"]
            landmark = landmarks_rearrange(landmark)
            landmarks.append(landmark)

        return images, landmarks

    def __getitem__(self, index: int) -> Tuple[list[np.ndarray], list[np.ndarray], str]:
        image = self.data[index]
        landmarks = self.targets[index]
        filename = self.image_filenames[index]

        if self.transforms is not None:
            image, landmarks = self.transforms(image, landmarks)

        return image, landmarks, filename

    def __getitems__(
            self, indices: list[int]
    ) -> list[Tuple[list[np.ndarray], list[np.ndarray], str]]:
        return list(map(lambda x: self.__getitem__(x), indices))

    def __len__(self):
        return len(self.data)
