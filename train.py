import os
import sys
from datetime import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader

from cobb_angle.data import BaseDataset
from cobb_angle.loss import LossAll
from cobb_angle.model import BigBrainNet

os.environ["TORCH_HOME"] = "./weights"


def collater(data):
    out_data_dict = {}
    for name in data[0]:
        out_data_dict[name] = []
    for sample in data:
        for name in sample:
            out_data_dict[name].append(torch.from_numpy(sample[name]))
    for name in out_data_dict:
        out_data_dict[name] = torch.stack(out_data_dict[name], dim=0)
    return out_data_dict


train_dataset = BaseDataset("./data", phase="train")
test_dataset = BaseDataset("./data", phase="test")

model = BigBrainNet(weights="DEFAULT")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6, last_epoch=-1)
criterion = LossAll()

train1_loader = DataLoader(train_dataset, batch_size=2)
train2_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collater)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collater)
now = datetime.now().strftime("%Y_%m%d_%H%M%S")
save_path = f"big_brain_net_{now}.pt"
