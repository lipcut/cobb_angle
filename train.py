import logging
import os
from datetime import datetime

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from cobb_angle.data import LandmarkDataset
from cobb_angle.loss import WingLossWithRegularization
from cobb_angle.models.cpn import (CascadedPyramidNetwork,
                                   CascadedPyramidNetworkConfig)
from cobb_angle.transform import \
    spine_dataset16_test_transforms as test_transforms
from cobb_angle.transform import \
    spine_dataset16_train_transforms as train_trasnforms
from cobb_angle.transform import \
    spine_dataset16_val_transforms as val_transforms

os.environ["TORCH_HOME"] = "./weights"


train_dataset = LandmarkDataset(root="./data")
test_dataset = LandmarkDataset(root="./data", train=False, transforms=test_transforms)

train_dataset, val_dataset = random_split(train_dataset, (0.8, 0.2))

config = CascadedPyramidNetworkConfig()
model = CascadedPyramidNetwork(config)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=1.25e-4)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=-1)
criterion = WingLossWithRegularization()

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

now = datetime.now().strftime("%Y_%m%d_%H%M%S")
save_file = f"bigbrain_net_{now}.pt"
save_dir = "weights"

logging.basicConfig(
    filename=os.path.join(save_dir, "loss.txt"), level=logging.DEBUG, format=""
)

EPOCH = 5


def train():
    for epoch in range(1, EPOCH + 1):
        print(f"Epoch {epoch}/{EPOCH}")
        print("-" * 10)

        train_loss = []
        train_dataset.dataset.transforms = train_trasnforms
        for data, targets, image_filename in train_loader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            model.train()
            predictions = model(data)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())

        val_loss = []
        val_dataset.dataset.transforms = val_transforms
        for data, targets, image_filename in val_loader:
            data, targets = data.to(device), targets.to(device)
            model.eval()
            with torch.no_grad():
                predictions = model(data)
                loss = criterion(predictions, targets)
                val_loss.append(loss.item())

        results = (
            f"Train Loss: {sum(train_loss)/len(train_loss):.2f} "
            f"Val Loss: {sum(val_loss)/len(val_loss):.2f}"
            "\n"
        )

        logging.debug(results)
        print(results)


if __name__ == "__main__":
    train()
