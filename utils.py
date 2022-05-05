from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    epochs: int,
    device: torch.device,
    save_dir: Path,
    log_interval=200,
):
    print("Training started...")
    model.train()
    losses = []

    for epoch in range(1, epochs + 1):
        epoch_loss = 0

        for batch_idx, (images, captions) in tqdm(enumerate(train_loader), leave=False):
            optimizer.zero_grad()

            images, captions = images.to(device), captions.to(device)
            outputs = model(images, captions[:-1])

            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_idx += 1
            if batch_idx % log_interval == 0:
                print(
                    f"Epoch: {epoch}\t"
                    f"[{batch_idx * len(images)}/{len(train_loader.dataset)}"
                    f" ({100. * batch_idx / len(train_loader):.0f}%)]\t"
                    f"Loss: {loss.item() / len(images):.6f}"
                )

        losses.append(epoch_loss / len(train_loader.dataset))

    torch.save(
        model.state_dict(),
        save_dir / "autoencodercnn_epoch_1.pth",
    )

    return losses


@torch.no_grad()
def valid(model, valid_loader, criterion, device):
    model.eval()

    # valid_loss = 0
    captions = []

    for images, captions in tqdm(enumerate(valid_loader), leave=False):
        images, captions = images.to(device), captions.to(device)

        # outputs = model(images, captions[:-1])
        # loss = criterion(denoised, pure)
        # valid_loss += loss.item()

    #         for n, d in zip(noisy, denoised):
    #             noisy_imgs.append(n.cpu())
    #             denoised_imgs.append(d.cpu())

    # valid_loss /= len(valid_loader.dataset)
    # print(f"Average test loss: {valid_loss:.6f}")

    return captions
