"""Train a global MNIST autoencoder and save checkpoint."""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.Autoencoder import AutoEncoder, train_model


def parse_args():
    parser = argparse.ArgumentParser(description="Train a global MNIST autoencoder")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="./weights/mnist")
    return parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = AutoEncoder()
    print("Training global MNIST AE")
    train_model(model, train_loader, num_epochs=args.epochs, device=device, lr=args.lr)

    save_path = out_dir / "ae_global.pt"
    torch.save(model.state_dict(), save_path)
    print(f"Saved global AE checkpoint to: {save_path}")


if __name__ == "__main__":
    main()
