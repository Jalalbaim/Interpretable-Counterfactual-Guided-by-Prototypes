"""Train 10 class-specific MNIST autoencoders and save checkpoints."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models.Autoencoder import AutoEncoder


def parse_args():
    parser = argparse.ArgumentParser(description="Train class-specific MNIST autoencoders")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./weights")
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_class(model, loader, device, epochs, lr):
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.6f}")


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./", train=True, download=True, transform=transform)

    labels = train_dataset.targets
    for digit in range(10):
        print(f"Training class-specific AE for digit {digit}")
        indices = (labels == digit).nonzero(as_tuple=False).squeeze(1).tolist()
        class_dataset = Subset(train_dataset, indices)
        class_loader = DataLoader(class_dataset, batch_size=args.batch_size, shuffle=True)

        model = AutoEncoder()
        print(f"\nTraining class-specific AE for digit {digit} (samples={len(class_dataset)})")
        train_single_class(model, class_loader, device, args.epochs, args.lr)

        save_path = out_dir / f"ae_class_{digit}.pt"
        torch.save(model.state_dict(), save_path)
        print(f"Saved class {digit} AE checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
