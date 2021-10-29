#  ------------------------------------------------------------------------------------------
# Adapted from the example at https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html
#  ------------------------------------------------------------------------------------------
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter


def main() -> None:
    log_dir = Path("outputs")
    log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    x = torch.arange(-20, 20, 0.1).view(-1, 1)
    y = -2 * x + 0.1 * torch.randn(x.size())

    model = nn.Linear(1, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(10):
        y1 = model(x)
        loss = criterion(y1, y)
        writer.add_scalar("Loss/train", loss, epoch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    writer.flush()


if __name__ == "__main__":
    main()
