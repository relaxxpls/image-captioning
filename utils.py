from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import EncoderDecoderModel
from vocabulary import CocoCaptionsVocabulary


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

        for batch_idx, (images, captions) in tqdm(
            enumerate(train_loader), leave=False, total=len(train_loader)
        ):
            optimizer.zero_grad()

            images, captions = images.to(device), captions.to(device)
            outputs = model(images, captions[:, :-1])
            outputs = outputs.view(outputs.shape[0], outputs.shape[2], outputs.shape[1])

            loss = criterion(outputs, captions)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            batch_idx += 1
            if batch_idx % log_interval == 0:
                tqdm.write(
                    f"Epoch {epoch}/{epochs}"
                    f"({100.0 * batch_idx / len(train_loader):.1f}% done)\t| "
                    f"Loss: {loss.item():.6f}"
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

    valid_loss = 0
    captions = []

    for batch_idx, (images, captions) in tqdm(enumerate(valid_loader), leave=False):
        images, captions = images.to(device), captions.to(device)

        # outputs = model(images, captions[:, :-1])
        # captions_predicted = [predict(images[i]) for ]
        # outputs = outputs.view(outputs.shape[0], outputs.shape[2], outputs.shape[1])
        # loss = criterion(outputs, captions)
        # loss = criterion(denoised, pure)
        # valid_loss += loss.item()

    valid_loss /= len(valid_loader)
    print(f"Valid loss: {valid_loss:.4f}")

    return captions


@torch.no_grad()
def predict(
    model: EncoderDecoderModel,
    image: torch.Tensor,
    vocabulary: CocoCaptionsVocabulary,
    max_length=20,
    device="cpu",
):
    result = []
    states: Tuple[torch.Tensor, torch.Tensor] = None

    image = image.unsqueeze(0).to(device)
    x = model.encoderCNN(image)
    x = x.unsqueeze(1)

    for _ in range(max_length):
        hiddens, states = model.decoderRNN.lstm(x, states)
        hiddens = hiddens.squeeze(1)
        outputs = model.decoderRNN.linear(hiddens)

        predicted = outputs.argmax(1)
        x = model.decoderRNN.embed(predicted)
        x = x.unsqueeze(1)

        predicted = predicted.item()
        predicted_word = vocabulary.idx2word[predicted]
        result.append(predicted_word)

        if predicted == vocabulary.eos:
            break

    result = " ".join(result)

    return result
