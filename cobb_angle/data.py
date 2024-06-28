import os

import cv2
from scipy.io import loadmat
from torch.utils import data

from . import pre_proc


class BaseDataset(data.Dataset):
    def __init__(self, data_dir, phase, input_h=1024, input_w=512, down_ratio=4):
        super(BaseDataset, self).__init__()
        self.data_dir = data_dir
        self.phase = phase
        self.input_h = input_h
        self.input_w = input_w
        self.down_ratio = down_ratio
        self.class_name = ["__background__", "cell"]
        self.num_classes = 68
        self.img_dir = os.path.join(data_dir, "data", self.phase)
        self.img_ids = sorted(os.listdir(self.img_dir))

    def load_annotation(self, index):
        img_id = self.img_ids[index]
        annotation_dir = os.path.join(
            self.data_dir, "labels", self.phase, img_id + ".mat"
        )
        pts = loadmat(annotation_dir)["p2"]
        return pts

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        image = cv2.imread(os.path.join(self.img_dir, img_id))
        if self.phase == "test":
            images = pre_proc.processing_test(
                image=image, input_h=self.input_h, input_w=self.input_w
            )
            return {"images": images, "img_id": img_id}
        else:
            aug_label = False
            if self.phase == "train":
                aug_label = True
            pts = self.load_annotation(index)  # num_obj x h x w
            out_image, pts_2 = pre_proc.processing_train(
                image=image,
                pts=pts,
                image_h=self.input_h,
                image_w=self.input_w,
                down_ratio=self.down_ratio,
                aug_label=aug_label,
                img_id=img_id,
            )

            data_dict = pre_proc.generate_ground_truth(
                image=out_image,
                pts_2=pts_2,
                image_h=self.input_h // self.down_ratio,
                image_w=self.input_w // self.down_ratio,
                img_id=img_id,
            )
            return data_dict

    def __len__(self):
        return len(self.img_ids)
