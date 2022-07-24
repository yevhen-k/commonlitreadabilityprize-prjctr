import torch
import numpy as np
import os
import random

__all__ = ['set_seed', 'train_loop', 'test_loop']


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train_loop(dataloader: torch.utils.data.DataLoader, model: torch.nn, loss_fn: torch.nn.MSELoss, optimizer: torch.optim.Optimizer) -> None:
    model.train()
    size = len(dataloader.dataset)
    for batch, (y, X) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        RMSE_loss = torch.sqrt(loss_fn(pred, y["label"]))

        # Backpropagation
        optimizer.zero_grad()
        RMSE_loss.backward()
        optimizer.step()

        if batch % 10 == 0:
            loss, current = RMSE_loss.item(), batch * len(y["label"])
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader: torch.utils.data.DataLoader, model: torch.nn, loss_fn: torch.nn.MSELoss) -> float:
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0

    with torch.no_grad():
        for y, X in dataloader:
            pred = model(X)
            test_loss += torch.sqrt(loss_fn(pred, y["label"])).item()

    test_loss /= num_batches
    print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")
    return test_loss
