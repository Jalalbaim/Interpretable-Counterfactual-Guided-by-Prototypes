from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
from torchvision import datasets, transforms
from tqdm import tqdm

from algorithm import Counterfactuals
from models.Model_MNIST import Model
from utils.ae_io import load_ae

MODE = "original"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Saliency-based pixel removal followed by prototype-guided recovery (MNIST)."
    )
    parser.add_argument("--index", type=int, default=6, help="MNIST test index")
    parser.add_argument("--top-k", type=int, default=200, help="Number of highest-saliency pixels to remove")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"), help="Directory with checkpoints")
    parser.add_argument("--output", type=Path, default=Path(f"outputs/pixel_selection_panel_{MODE}.png"), help="Path to save 4-panel figure")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-proto", type=int, default=15, help="Number of nearest neighbors to build prototype")
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--theta", type=float, default=200.0)
    parser.add_argument("--gamma", type=float, default=100.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--cap", type=float, default=0.0)
    return parser.parse_args()


def set_determinism(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_project_components(device: torch.device, weights_dir: Path) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    model = Model().to(device)
    model_path = weights_dir / "mnist_cgp_model_weights.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    ae_all = load_ae(weights_dir / "autoencoder_mnist.pth", device)
    encoder = ae_all.encoder.to(device).eval()
    return model, ae_all, encoder


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def compute_saliency_map(model: torch.nn.Module, x: Tensor, class_idx: int) -> Tensor:
    x_for_grad = x.detach().clone().requires_grad_(True)
    logits = model(x_for_grad)
    score = logits[:, class_idx].sum()
    score.backward()
    saliency = x_for_grad.grad.detach().abs()
    return saliency


def perturb_topk_pixels(x: Tensor, saliency: Tensor, top_k: int, black_value: float = 0.0) -> tuple[Tensor, Tensor]:
    flat_sal = saliency.view(-1)
    k = min(top_k, flat_sal.numel())
    topk_idx = torch.topk(flat_sal, k=k, largest=True).indices

    x_perturbed = x.detach().clone()
    flat_img = x_perturbed.view(-1)
    flat_img[topk_idx] = black_value

    mask = torch.zeros_like(flat_sal, dtype=torch.bool)
    mask[topk_idx] = True
    return x_perturbed, mask.view_as(x)


def find_closest_prototype(
    encoder: torch.nn.Module,
    x_ref: Tensor,
    class_samples: Tensor,
    k_proto: int,
) -> Tensor:
    with torch.no_grad():
        z_ref = encoder(x_ref)
        z_samples = encoder(class_samples)
        # Flatten only for distance computation
        z_ref_flat = z_ref.flatten(1)
        z_samples_flat = z_samples.flatten(1)
        dists = torch.cdist(z_ref_flat, z_samples_flat).squeeze(0)
        nn_idx = torch.topk(dists, k=min(k_proto, len(dists)), largest=False).indices
        # Average in original (unflattened) shape so it matches loss_proto
        proto = z_samples[nn_idx].mean(dim=0, keepdim=True)
    return proto


def recover_to_original_class(
    cf_algo: Counterfactuals,
    x_start: Tensor,
    target_class: int,
    target_proto: Tensor,
    *,
    c: float,
    beta: float,
    theta: float,
    cap: float,
    gamma: float,
    max_iterations: int,
    lr: float,
) -> Tensor:
    perturbation = torch.zeros_like(x_start, requires_grad=True)
    optimizer = torch.optim.Adam([perturbation], lr=lr)

    for _ in tqdm(range(max_iterations), desc="Recovering"):
        optimizer.zero_grad()
        x_candidate = torch.clamp(x_start + perturbation, 0.0, 1.0)

        logits = cf_algo.model(x_candidate)
        target_logit = logits[:, target_class]
        mask = torch.ones(logits.shape[1], dtype=torch.bool, device=logits.device)
        mask[target_class] = False
        max_other = logits[:, mask].max(dim=1).values
        l_pred = torch.clamp(max_other - target_logit, min=-cap).mean()

        l1, l2 = cf_algo.loss_l1_l2(perturbation)
        l_ae = cf_algo.loss_ae(x_candidate, gamma=gamma)
        l_proto = cf_algo.loss_proto(x_candidate, target_proto, theta=theta)
        loss = cf_algo.total_Loss(c, l_pred, beta, l1, l2, l_ae, l_proto)

        loss.backward()
        torch.nn.utils.clip_grad_norm_([perturbation], max_norm=1.0)
        optimizer.step()

    with torch.no_grad():
        return torch.clamp(x_start + perturbation, 0.0, 1.0)

# add a function recover to the closest class
def recover_to_closest_class(
    cf_algo: Counterfactuals,
    x_start: Tensor,
    target_class: int,
    target_proto: Tensor,
    *,
    c: float,
    beta: float,
    theta: float,
    cap: float,
    gamma: float,
    max_iterations: int,
    lr: float,
) -> Tensor:
    perturbation = torch.zeros_like(x_start, requires_grad=True)
    optimizer = torch.optim.Adam([perturbation], lr=lr)

    for _ in tqdm(range(max_iterations), desc="Recovering"):
        optimizer.zero_grad()
        x_candidate = torch.clamp(x_start + perturbation, 0.0, 1.0)

        logits = cf_algo.model(x_candidate)
        target_logit = logits[:, target_class]
        mask = torch.ones(logits.shape[1], dtype=torch.bool, device=logits.device)
        mask[target_class] = False
        max_other = logits[:, mask].max(dim=1).values
        l_pred = torch.clamp(max_other - target_logit, min=-cap).mean()

        l1, l2 = cf_algo.loss_l1_l2(perturbation)
        l_ae = cf_algo.loss_ae(x_candidate, gamma=gamma)
        l_proto = cf_algo.loss_proto(x_candidate, target_proto, theta=theta)
        loss = cf_algo.total_Loss(c, l_pred, beta, l1, l2, l_ae, l_proto)

        loss.backward()
        torch.nn.utils.clip_grad_norm_([perturbation], max_norm=1.0)
        optimizer.step()

    with torch.no_grad():
        return torch.clamp(x_start + perturbation, 0.0, 1.0)

def main() -> None:
    args = parse_args()
    set_determinism(args.seed)
    device = resolve_device(args.device)

    model, ae_all, encoder = load_project_components(device, args.weights_dir)
    model.eval()
    ae_all.eval()
    encoder.eval()

    test_ds = datasets.MNIST(root="./", train=False, download=True, transform=transforms.ToTensor())
    train_ds = datasets.MNIST(root="./", train=True, download=True, transform=transforms.ToTensor())

    x_orig, _ = test_ds[args.index]
    x_orig = x_orig.unsqueeze(0).to(device)

    with torch.no_grad():
        orig_class = int(model(x_orig).argmax(dim=1).item())

    saliency = compute_saliency_map(model, x_orig, orig_class)
    x_perturbed, _ = perturb_topk_pixels(x_orig, saliency, top_k=args.top_k, black_value=0.0)

    train_imgs = torch.stack([x for x, _ in train_ds], dim=0).to(device)
    with torch.no_grad():
        batch_size = 256
        preds = []
        for i in range(0, train_imgs.size(0), batch_size):
            preds.append(model(train_imgs[i : i + batch_size]).argmax(dim=1))
        train_preds = torch.cat(preds)

    class_samples = train_imgs[train_preds == orig_class]
    closest_proto = find_closest_prototype(encoder, x_perturbed, class_samples, args.k_proto)

    cf_algo = Counterfactuals(model, encoder, ae_all, device=device)

    if MODE == "original":
        x_recovered = recover_to_original_class(
            cf_algo,
            x_perturbed,
            orig_class,
            closest_proto,
            c=args.c,
            beta=args.beta,
            theta=args.theta,
            cap=args.cap,
            gamma=args.gamma,
            max_iterations=args.max_iterations,
            lr=args.lr,
        )
    elif MODE == "closest":
        x_recovered = recover_to_closest_class(
            cf_algo,
            x_perturbed,
            orig_class,
            closest_proto,
            c=args.c,
            beta=args.beta,
            theta=args.theta,
            cap=args.cap,
            gamma=args.gamma,
            max_iterations=args.max_iterations,
            lr=args.lr,
        )

    with torch.no_grad():
        perturbed_class = int(model(x_perturbed).argmax(dim=1).item())
        recovered_class = int(model(x_recovered).argmax(dim=1).item())

    args.output.parent.mkdir(parents=True, exist_ok=True)

    saliency_np = saliency.detach().cpu().squeeze().numpy()
    saliency_np = saliency_np / (saliency_np.max() + 1e-8)

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    axes[0].imshow(x_orig.detach().cpu().squeeze().numpy(), cmap="gray")
    axes[0].set_title(f"Original\nclass={orig_class}")
    axes[0].axis("off")

    axes[1].imshow(saliency_np, cmap="hot")
    axes[1].set_title("Saliency")
    axes[1].axis("off")

    axes[2].imshow(x_perturbed.detach().cpu().squeeze().numpy(), cmap="gray")
    axes[2].set_title(f"Perturbed\nclass={perturbed_class}")
    axes[2].axis("off")

    axes[3].imshow(x_recovered.detach().cpu().squeeze().numpy(), cmap="gray")
    axes[3].set_title(f"Recovered\nclass={recovered_class}")
    axes[3].axis("off")

    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    plt.close(fig)

    print(f"Saved panel to {args.output}")


if __name__ == "__main__":
    main()
