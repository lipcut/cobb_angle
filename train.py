import os
from datetime import datetime

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from cobb_angle.data import BaseDataset
from cobb_angle.loss import LossAll
from cobb_angle.model import BigBrainNet

os.environ["TORCH_HOME"] = "./weights"


def collater(data):
    return {
        key: torch.tensor(np.array([datum[key] for datum in data]))
        for key in data[0].keys()
    }


train_dataset = BaseDataset("./data", phase="train")
test_dataset = BaseDataset("./data", phase="test")

model = BigBrainNet(weights="DEFAULT")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.6, last_epoch=-1)
criterion = LossAll()

train_dataset, val_dataset = random_split(train_dataset, (0.8, 0.2))
train_loader = DataLoader(train_dataset, batch_size=2, collate_fn=collater)
val_loader = DataLoader(val_dataset, batch_size=2, collate_fn=collater)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, collate_fn=collater)
now = datetime.now().strftime("%Y_%m%d_%H%M%S")
save_path = f"bigbrain_net_{now}.pt"
save_dir = "weights"

for epoch in range(1, 25):
    train_loss = []
    for data in tqdm(train_loader):
        data = {key: value.to(device) for key, value in data.items()}
        optimizer.zero_grad()
        model.train()
        predictions = model(data["input"])
        loss = criterion(predictions, data)
        loss.backward()
        optimizer.step()
        train_loss.append(loss)

    val_loss = []
    for data in tqdm(val_loader):
        data = {key: value.to(device) for key, value in data.items()}
        model.eval()
        predictions = model(data["input"])
        loss = criterion(predictions, data)
        val_loss.append(loss)

    print(
        f"Epoch: {epoch}, "
        f"Train Loss: {train_loss/len(train_loss):.2f} "
        f"Val Loss: {val_loss/len(val_loss):.2f}"
    )
