from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import Tensor
from torchvision import datasets, transforms
from tqdm import tqdm

from algorithm import Counterfactuals
from models.Model_MNIST import Model
from utils.ae_io import load_ae



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latent-space view for original/perturbed/recovered/prototype.")
    parser.add_argument("--index", type=int, default=0, help="MNIST test index")
    parser.add_argument("--top-k", type=int, default=60, help="Number of highest-saliency pixels to black out")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--output", type=Path, default=Path("outputs/latent_perturbed.png"))
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k-proto", type=int, default=15)
    parser.add_argument("--subset", type=int, default=4000, help="Train samples used for background latent cloud")
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
    logits[:, class_idx].sum().backward()
    return x_for_grad.grad.detach().abs()


def perturb_topk_pixels(x: Tensor, saliency: Tensor, top_k: int) -> Tensor:
    flat_sal = saliency.view(-1)
    k = min(top_k, flat_sal.numel())
    topk_idx = torch.topk(flat_sal, k=k, largest=True).indices

    x_perturbed = x.detach().clone()
    x_perturbed.view(-1)[topk_idx] = 0.0
    return x_perturbed


def find_closest_prototype(encoder: torch.nn.Module, x_ref: Tensor, class_samples: Tensor, k_proto: int) -> Tensor:
    with torch.no_grad():
        z_ref = encoder(x_ref).flatten(1)
        z_samples = encoder(class_samples).flatten(1)
        dists = torch.cdist(z_ref, z_samples).squeeze(0)
        nn_idx = torch.topk(dists, k=min(k_proto, len(dists)), largest=False).indices
        return z_samples[nn_idx].mean(dim=0, keepdim=True)


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


def main() -> None:
    args = parse_args()
    set_determinism(args.seed)
    device = resolve_device(args.device)

    model, ae_all, encoder = load_project_components(device, args.weights_dir)
    model.eval()
    ae_all.eval()
    encoder.eval()

    train_ds = datasets.MNIST(root="./", train=True, download=True, transform=transforms.ToTensor())
    test_ds = datasets.MNIST(root="./", train=False, download=True, transform=transforms.ToTensor())

    x_orig, _ = test_ds[args.index]
    x_orig = x_orig.unsqueeze(0).to(device)

    with torch.no_grad():
        orig_class = int(model(x_orig).argmax(dim=1).item())

    saliency = compute_saliency_map(model, x_orig, orig_class)
    x_perturbed = perturb_topk_pixels(x_orig, saliency, top_k=args.top_k)

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

    subset_n = min(args.subset, train_imgs.size(0))
    background = train_imgs[:subset_n]

    with torch.no_grad():
        z_bg = encoder(background).flatten(1).cpu().numpy()
        y_bg = train_preds[:subset_n].cpu().numpy()
        z_orig = encoder(x_orig).flatten(1).cpu().numpy()
        z_pert = encoder(x_perturbed).flatten(1).cpu().numpy()
        z_rec = encoder(x_recovered).flatten(1).cpu().numpy()
        z_proto = closest_proto.cpu().numpy()

    all_latent = np.concatenate([z_bg, z_orig, z_pert, z_rec, z_proto], axis=0)
    pca = PCA(n_components=2, random_state=args.seed)
    proj = pca.fit_transform(all_latent)

    bg_proj = proj[:subset_n]
    orig_proj = proj[subset_n]
    pert_proj = proj[subset_n + 1]
    rec_proj = proj[subset_n + 2]
    proto_proj = proj[subset_n + 3]

    args.output.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(bg_proj[:, 0], bg_proj[:, 1], c=y_bg, cmap="tab10", alpha=0.25, s=10)
    plt.scatter(orig_proj[0], orig_proj[1], c="black", marker="o", s=160, label="Original")
    plt.scatter(pert_proj[0], pert_proj[1], c="red", marker="x", s=180, label="Perturbed")
    plt.scatter(rec_proj[0], rec_proj[1], c="limegreen", marker="*", s=220, label="Recovered")
    plt.scatter(proto_proj[0], proto_proj[1], c="blue", marker="D", s=150, label="Closest prototype")

    plt.annotate("", xy=(pert_proj[0], pert_proj[1]), xytext=(orig_proj[0], orig_proj[1]), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "red"})
    plt.annotate("", xy=(rec_proj[0], rec_proj[1]), xytext=(pert_proj[0], pert_proj[1]), arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "green"})
    plt.annotate("", xy=(proto_proj[0], proto_proj[1]), xytext=(rec_proj[0], rec_proj[1]), arrowprops={"arrowstyle": "->", "lw": 1.0, "color": "blue"})

    plt.title("Latent space (PCA): original vs perturbed vs recovered vs prototype")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.colorbar(sc, label="Predicted class (train subset)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)
    plt.close()

    with torch.no_grad():
        pert_class = int(model(x_perturbed).argmax(dim=1).item())
        rec_class = int(model(x_recovered).argmax(dim=1).item())

    print(f"Saved latent visualization to {args.output}")
    print(f"Classes -> original: {orig_class}, perturbed: {pert_class}, recovered: {rec_class}")


if __name__ == "__main__":
    main()
