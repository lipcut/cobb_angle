import logging
import os
from pathlib import Path

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split

from cobb_angle.data import LandmarkDataset
from cobb_angle.loss import WingLossWithRegularization
from cobb_angle.model import CascadedPyramidNetwork, CascadedPyramidNetworkConfig
from cobb_angle.transform import spine_dataset16_test_transforms as test_transforms
from cobb_angle.transform import spine_dataset16_train_transforms as train_trasnforms
from cobb_angle.dsnt import dsnt


os.environ["TORCH_HOME"] = "./weights"


train_dataset = LandmarkDataset(root="./data")
test_dataset = LandmarkDataset(root="./data", train=False, transforms=test_transforms)

train_dataset, val_dataset = random_split(train_dataset, (0.8, 0.2))

config = CascadedPyramidNetworkConfig()
model = CascadedPyramidNetwork(config)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-3)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.96, last_epoch=-1)
criterion = WingLossWithRegularization()

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1)

save_file = "bigbrain_net.pt"
save_dir = "weights"
logging_file = Path(save_dir, "loss.txt")

if not logging_file.exists():
    logging_file.parent.mkdir(parents=True, exist_ok=True)
    with open(logging_file, "w") as f:
        f.write("train_loss,val_loss\n")

logging.basicConfig(
    filename=os.path.join(save_dir, "loss.txt"), level=logging.DEBUG, format=""
)

EPOCH = 25


def train():
    for epoch in range(1, EPOCH + 1):
        print(f"Epoch {epoch}/{EPOCH}")
        print("-" * 10)

        best_val_loss = 1e5

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
        val_dataset.dataset.transforms = test_transforms
        for data, targets, image_filename in val_loader:
            data, targets = data.to(device), targets.to(device)
            model.eval()
            with torch.no_grad():
                predictions = model(data)
                loss = criterion(predictions, targets)
                val_loss.append(loss.item())

        mean_train_loss = sum(train_loss) / len(train_loss)
        mean_val_loss = sum(val_loss) / len(val_loss)

        if epoch % 5 == 0:
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"epoch_{epoch}_" + save_file),
            )

        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, "best_" + save_file),
            )

        print(
            f"Train Loss: {mean_train_loss:.2f} " f"Val Loss: {mean_val_loss:.2f}" "\n"
        )

        logging.debug(f"{mean_train_loss},{mean_val_loss}")


def eval():
    config = CascadedPyramidNetworkConfig()
    model = CascadedPyramidNetwork(config)
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_bigbrain_net.pt")))
    model = model.to(device)

    l2_norm = []
    for data, targets, image_filename in test_loader:
        model.eval()
        with torch.no_grad():
            targets = targets.to(device)
            data = data.to(device)

            predictions = model(data)
            stage2 = predictions[1]
            batch_size, channels, height, width = stage2.shape
            predicted_landmarks = torch.zeros_like(targets)
            predicted_landmarks[..., 0] = (dsnt(stage2)[..., 0] + 1) * (width - 1) / 2
            predicted_landmarks[..., 1] = (dsnt(stage2)[..., 1] + 1) * (height - 1) / 2
            l2_norm.append(torch.norm(predicted_landmarks - targets, dim=-1).mean())

    print(f"Eucliean distance for landmarks: {sum(l2_norm) / len(l2_norm)}")


if __name__ == "__main__":
    train()
    # eval()
