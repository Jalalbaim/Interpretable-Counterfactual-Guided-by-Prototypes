"""Checkpoint helpers for MNIST autoencoders."""

from pathlib import Path

import torch

from models.Autoencoder import AutoEncoder


def _to_device(device):
    return torch.device(device) if not isinstance(device, torch.device) else device


def load_ae(checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {checkpoint_path}")

    model = AutoEncoder()
    state_dict = torch.load(checkpoint_path, map_location=_to_device(device))
    model.load_state_dict(state_dict)
    model.to(_to_device(device))
    model.eval()
    return model


def load_class_aes(checkpoint_dir, device):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Class AE checkpoint directory not found: {checkpoint_dir}")

    class_aes = {}
    missing = []
    for digit in range(10):
        ckpt = checkpoint_dir / f"ae_class_{digit}.pt"
        if not ckpt.exists():
            missing.append(str(ckpt))
            continue
        class_aes[digit] = load_ae(ckpt, device)

    if missing:
        missing_list = "\n".join(missing)
        raise FileNotFoundError(
            "Missing class-specific AE checkpoints:\n"
            f"{missing_list}"
        )

    return class_aes
