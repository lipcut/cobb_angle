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
        self.data, self.targets = self._load_data()

    @property
    def image_folder(self) -> str:
        return os.path.join(self.root, "data", self.phase)

    @property
    def landmarks_folder(self) -> str:
        return os.path.join(self.root, "labels", self.phase)

    def _load_data(self) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
        return images, landmarks

    def __getitem__(self, index: int) -> Tuple[List[np.ndarray], List[np.ndarray], str]:
        image = self.data[index]
        landmarks = self.targets[index]
        image_filename = self.image_filenames[index]

        if self.transforms is not None:
            image, landmarks = self.transforms(image, landmarks)

        return image, landmarks, image_filename

    def __getitems__(
        self, indices: List
    ) -> List[Tuple[List[np.ndarray], List[np.ndarray], str]]:
        image_list = [self.data[index] for index in indices]
        landmarks_list = [self.targets[index] for index in indices]
        image_filename_list = [self.image_filenames[index] for index in indices]

        return [
            (
                self.transforms(image, landmarks)
                if self.transforms is not None
                else (image, landmarks)
            )
            + (image_filename,)
            for image, landmarks, image_filename in zip(
                image_list, landmarks_list, image_filename_list
            )
        ]

    def __len__(self):
        return len(self.data)
