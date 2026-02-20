"""
Counterfactuals Guided By Prototypes — CIFAR-10
Adapted from CFG_main.py (MNIST) for CIFAR-10 images (3×32×32).
@author: BAIM M.J
"""

import json
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

from algorithm import Counterfactuals
from metrics.IM1_cifar import compute_im1
from metrics.IM2_cifar import compute_im2
from models.AE_cifar import Autoencoder
from models.Model_ciafr10 import get_model
from utils.ae_io_cifar import load_ae, load_class_aes

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROTO_METHOD = "kmeans"  # None or "kmeans"
K_CLUSTERS = 3
CLASSIFIER_NAME = "resnet18"  # "resnet18" or "resnet50"

AE_CHECKPOINT_DIR = "./weights"
AE_GLOBAL_CHECKPOINT = "./weights/AE_cifar10"

CIFAR10_CLASSES = [
    "plane", "car", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trained_model(model, autoencoder, weights_classif, weights_auto, device):
    classif = torch.load(weights_classif, map_location=device)
    model.load_state_dict(classif)
    model.to(device)

    autoenc = torch.load(weights_auto, map_location=device)
    autoencoder.load_state_dict(autoenc)
    autoencoder.to(device)

    encoder = autoencoder.encoder

    return model, autoencoder, encoder


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs("./outputs", exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    # Date / time for file naming
    time_str = datetime.now().strftime("%H-%M-%S")
    date_str = datetime.now().strftime("%Y-%m-%d") + f"_{time_str}"

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    transform = transforms.Compose([transforms.ToTensor()])

    train_dataset = datasets.CIFAR10(root="./", train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root="./", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    print(len(images))
    index = random.randint(0, len(images) - 1)
    print(f"Selected index: {index}")

    log_name = f"./logs/{date_str}_CIFAR10_{index}_{PROTO_METHOD}_K{K_CLUSTERS}"
    writer = SummaryWriter(log_dir=log_name)

    x_orig = images[index].unsqueeze(0).to(device)
    train_data = torch.stack([img for img, _ in train_loader.dataset], dim=0).to(device)

    # ------------------------------------------------------------------
    # Model & Autoencoder
    # ------------------------------------------------------------------
    model = get_model(device, name=CLASSIFIER_NAME)
    autoencoder = Autoencoder()

    if CLASSIFIER_NAME == "resnet50":
        model_weights = "./weights/resnet50_cifar10_final_model_epochs_60.pth"
    else:
        model_weights = "./weights/cifar10_resnet18.pth"

    autoencoder_weights = "./weights/AE_cifar10"

    model_trained, autoencoder_trained, encoder_trained = trained_model(
        model, autoencoder, model_weights, autoencoder_weights, device
    )
    print("Trained model, autoencoder, encoder loaded.")

    counterfactual = Counterfactuals(model_trained, encoder_trained, autoencoder_trained, device=device)

    # ------------------------------------------------------------------
    # Hyperparameters
    # ------------------------------------------------------------------
    run_hparams = {
        "c": 1.0,
        "beta": 0.1,
        "theta": 200,
        "cap": 0.01,
        "gamma": 100,
        "K": K_CLUSTERS,
        "lr": 1e-3,
        "max_iterations": 500,
        "method": PROTO_METHOD,
    }

    # ------------------------------------------------------------------
    # Generate counterfactual
    # ------------------------------------------------------------------
    print("Generating counterfactual...")
    x_counterfactual, details = counterfactual.algorithm_CGP(
        x_orig,
        train_data,
        c=run_hparams["c"],
        beta=run_hparams["beta"],
        theta=run_hparams["theta"],
        cap=run_hparams["cap"],
        gamma=run_hparams["gamma"],
        K=run_hparams["K"],
        max_iterations=run_hparams["max_iterations"],
        lr=run_hparams["lr"],
        proto_method=run_hparams["method"],
        device=device,
        writer=writer,
        return_details=True,
    )

    with torch.no_grad():
        pred_cf = model_trained(x_counterfactual).argmax(dim=1).item()
        pred_org = model_trained(x_orig).argmax(dim=1).item()

    # ------------------------------------------------------------------
    # IM1 / IM2 metrics
    # ------------------------------------------------------------------
    im1_value = None
    im2_value = None
    try:
        ae_all = load_ae(AE_GLOBAL_CHECKPOINT, device)
        class_aes = load_class_aes(AE_CHECKPOINT_DIR, device)
        im1_value = compute_im1(x_counterfactual, class_aes[pred_cf], class_aes[pred_org], reduction="mean").item()
        im2_value = compute_im2(x_counterfactual, class_aes[pred_cf], ae_all, reduction="mean").item()
        print(f"IM1: {im1_value:.6f} | IM2: {im2_value:.6f}")
    except FileNotFoundError as exc:
        print(f"Skipped IM1/IM2 computation: {exc}")

    # ------------------------------------------------------------------
    # Plot side by side (RGB)
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(ncols=2, figsize=(6, 3))

    orig_np = x_orig.detach().cpu().numpy()[0].transpose(1, 2, 0)
    axes[0].imshow(orig_np)
    axes[0].set_title(f"Original - {CIFAR10_CLASSES[pred_org]}")
    axes[0].axis("off")

    cf_np = x_counterfactual.detach().cpu().numpy()[0].transpose(1, 2, 0)
    axes[1].imshow(cf_np.clip(0, 1))
    axes[1].set_title(f"Counterfactual - {CIFAR10_CLASSES[pred_cf]}")
    axes[1].axis("off")

    filename_method = "nearest" if PROTO_METHOD is None else PROTO_METHOD
    image_path = f"./outputs/{date_str}_CIFAR10_{filename_method}_K{K_CLUSTERS}_{index}.png"
    plt.savefig(image_path)
    plt.close()

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    metadata = {
        "chosen_index": index,
        "original_predicted_class": details["orig_class"],
        "original_class_name": CIFAR10_CLASSES[details["orig_class"]],
        "counterfactual_predicted_class": details["cf_class"],
        "counterfactual_class_name": CIFAR10_CLASSES[details["cf_class"]],
        "target_class": details["target_class"],
        "target_class_name": CIFAR10_CLASSES[details["target_class"]],
        "classifier": CLASSIFIER_NAME,
        "hyperparameters": run_hparams,
        "iterations_until_found": details["final_iteration"],
        "final_loss": details["final_loss"],
        "im1": im1_value,
        "im2": im2_value,
        "saved_image": image_path,
    }

    metadata_path = f"./outputs/{date_str}_run_CIFAR10_{filename_method}_K{K_CLUSTERS}.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    writer.close()
    print(f"Saved figure to: {image_path}")
    print(f"Saved metadata to: {metadata_path}")


if __name__ == "__main__":
    main()
