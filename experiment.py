from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import datasets, transforms
from tqdm import tqdm


from models.Autoencoder import AutoEncoder
from models.Model_MNIST import Model
from utils.ae_io import load_ae, load_class_aes


METHODS = ["A", "B", "C", "D", "E", "F"]
EPS = 1e-6


@dataclass
class CFRunResult:
    x_cf: Tensor
    delta: Tensor
    num_grad_updates_used: int
    time_seconds: float
    original_class: int
    counterfactual_class: int


class FallbackCFGenerator:
    """Simple gradient-based counterfactual optimizer (Adam)."""

    def __init__(self, f_pred, ae_all, encoder, class_train_tensors: Dict[int, Tensor], device: torch.device):
        self.f_pred = f_pred
        self.ae_all = ae_all
        self.encoder = encoder
        self.class_train_tensors = class_train_tensors
        self.device = device

        self.class_train_encodings: Dict[int, Tensor] = {}
        self._precompute_class_encodings()

    @torch.no_grad()
    def _precompute_class_encodings(self) -> None:
        self.encoder.eval()
        for cls, data in self.class_train_tensors.items():
            if data.numel() == 0:
                self.class_train_encodings[cls] = torch.empty((0, 1), device=self.device)
                continue
            self.class_train_encodings[cls] = self.encoder(data.to(self.device)).flatten(1)

    @torch.no_grad()
    def _target_prototype(self, x0: Tensor, t0: int, K: int) -> Tuple[Tensor, int]:
        z0 = self.encoder(x0).flatten(1)
        prototypes = {}
        for cls in range(10):
            if cls == t0:
                continue
            z_cls = self.class_train_encodings[cls]
            dists = torch.cdist(z0, z_cls).squeeze(0)
            nn_idx = torch.topk(dists, k=min(K, len(dists)), largest=False).indices
            prototypes[cls] = z_cls[nn_idx].mean(dim=0, keepdim=True)

        best_cls = min(prototypes, key=lambda c: torch.norm(z0 - prototypes[c], p=2).item())
        return prototypes[best_cls], best_cls

    def _l_pred(self, x: Tensor, t0: int, kappa: float) -> Tensor:
        probs = torch.softmax(self.f_pred(x), dim=1)
        p_t0 = probs[:, t0]
        mask = torch.ones(probs.shape[1], dtype=torch.bool, device=probs.device)
        mask[t0] = False
        p_other = probs[:, mask].max(dim=1).values
        return torch.clamp(p_t0 - p_other, min=-kappa).mean()

    def generate(
        self,
        x0: Tensor,
        method: str,
        seed: int,
        c: float,
        kappa: float,
        beta: float,
        gamma: float,
        theta: float,
        K: int,
        max_updates: int,
        lr: float,
        clip_min: float,
        clip_max: float,
    ) -> CFRunResult:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.f_pred.eval()
        self.ae_all.eval()
        self.encoder.eval()

        with torch.no_grad():
            t0 = int(torch.argmax(self.f_pred(x0), dim=1).item())

        target_proto, _ = self._target_prototype(x0, t0=t0, K=K)

        delta = torch.zeros_like(x0, requires_grad=True)
        optimizer = torch.optim.Adam([delta], lr=lr)

        start = time.perf_counter()
        found_cf: Optional[Tensor] = None
        found_step: Optional[int] = None
        found_cls: Optional[int] = None

        for step in range(1, max_updates + 1):
            optimizer.zero_grad()
            x_cf = torch.clamp(x0 + delta, clip_min, clip_max)

            l1 = torch.norm(delta, p=1)
            l2 = torch.norm(delta, p=2) ** 2
            lae = gamma * torch.norm((x_cf - self.ae_all(x_cf)).flatten(1), p=2, dim=1).pow(2).mean()
            lproto = theta * torch.norm(self.encoder(x_cf).flatten(1) - target_proto, p=2, dim=1).pow(2).mean()
            lpred = self._l_pred(x_cf, t0=t0, kappa=kappa)

            if method == "A":
                loss = c * lpred + beta * l1 + l2
            elif method == "B":
                loss = c * lpred + beta * l1 + l2 + lae
            elif method == "C":
                loss = c * lpred + beta * l1 + l2 + lproto
            elif method == "D":
                loss = c * lpred + beta * l1 + l2 + lae + lproto
            elif method == "E":
                loss = beta * l1 + l2 + lproto
            elif method == "F":
                loss = beta * l1 + l2 + lae + lproto
            else:
                raise ValueError(f"Unknown method: {method}")

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                x_eval = torch.clamp(x0 + delta, clip_min, clip_max)
                cf_cls = int(torch.argmax(self.f_pred(x_eval), dim=1).item())
                if cf_cls != t0:
                    found_cf = x_eval.detach().clone()
                    found_step = step
                    found_cls = cf_cls
                    break

        elapsed = time.perf_counter() - start

        if found_cf is None:
            with torch.no_grad():
                found_cf = torch.clamp(x0 + delta, clip_min, clip_max).detach().clone()
                found_cls = int(torch.argmax(self.f_pred(found_cf), dim=1).item())
            found_step = max_updates

        return CFRunResult(
            x_cf=found_cf,
            delta=(found_cf - x0).detach(),
            num_grad_updates_used=int(found_step),
            time_seconds=float(elapsed),
            original_class=t0,
            counterfactual_class=int(found_cls),
        )


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_project_components(device: torch.device, ckpt_dir: Path) -> Tuple[torch.nn.Module, torch.nn.Module, Dict[int, torch.nn.Module], torch.nn.Module]:
    """Load classifier, global AE, class AEs, and encoder.

    Edit paths below if your checkpoints are elsewhere.
    """
    model = Model().to(device)
    model_path = ckpt_dir / "mnist_cgp_model_weights.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Classifier checkpoint not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    ae_all = load_ae(ckpt_dir / "autoencoder_mnist.pth", device)
    ae_class = load_class_aes(ckpt_dir, device)
    encoder = ae_all.encoder.to(device).eval()

    return model, ae_all, ae_class, encoder


def maybe_get_existing_generator(*_args, **_kwargs):
    """Hook for projects exposing a method-ID counterfactual generator.

    Return None by default in this repository.
    """
    return None


def sample_balanced_test_indices(labels: Tensor, samples_per_class: int, rng: np.random.Generator) -> List[int]:
    picks: List[int] = []
    labels_np = labels.cpu().numpy()
    for c in range(10):
        idx = np.where(labels_np == c)[0]
        if len(idx) < samples_per_class:
            raise ValueError(f"Class {c} has only {len(idx)} examples, need {samples_per_class}")
        picks.extend(rng.choice(idx, size=samples_per_class, replace=False).tolist())
    return picks


def ci95(std: float, n: int) -> float:
    return 1.96 * std / math.sqrt(max(n, 1))


def compute_im1_im2(x_cf: Tensor, t0: int, i_cf: int, ae_class: Dict[int, torch.nn.Module], ae_all: torch.nn.Module) -> Tuple[float, float]:
    with torch.no_grad():
        num_im1 = torch.norm((x_cf - ae_class[i_cf](x_cf)).flatten(1), p=2, dim=1).pow(2)
        den_im1 = torch.norm((x_cf - ae_class[t0](x_cf)).flatten(1), p=2, dim=1).pow(2) + EPS
        im1 = (num_im1 / den_im1).mean().item()

        num_im2 = torch.norm((ae_class[i_cf](x_cf) - ae_all(x_cf)).flatten(1), p=2, dim=1).pow(2)
        den_im2 = x_cf.flatten(1).abs().sum(dim=1) + EPS
        im2 = (num_im2 / den_im2).mean().item()
    return float(im1), float(im2)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, g in df.groupby("method"):
        n = len(g)
        vals = {
            "method": method,
            "n": n,
            "mean_time": g["time_seconds"].mean(),
            "std_time": g["time_seconds"].std(ddof=1),
            "mean_updates": g["num_grad_updates_used"].mean(),
            "std_updates": g["num_grad_updates_used"].std(ddof=1),
            "mean_im1": g["IM1"].mean(),
            "std_im1": g["IM1"].std(ddof=1),
            "mean_im2": g["IM2"].mean(),
            "std_im2": g["IM2"].std(ddof=1),
            "mean_en": g["EN"].mean(),
            "std_en": g["EN"].std(ddof=1),
        }
        vals["ci95_im1"] = ci95(vals["std_im1"], n)
        vals["ci95_im2"] = ci95(vals["std_im2"], n)
        vals["ci95_en"] = ci95(vals["std_en"], n)
        rows.append(vals)

    out = pd.DataFrame(rows).sort_values("method").reset_index(drop=True)
    return out


def make_figure(summary: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # (a) Time vs gradient updates with std error bars
    ax = axes[0]
    for _, r in summary.iterrows():
        ax.errorbar(
            r["mean_time"],
            r["mean_updates"],
            xerr=r["std_time"],
            yerr=r["std_updates"],
            fmt="o",
            capsize=3,
            label=r["method"],
        )
        ax.text(r["mean_time"], r["mean_updates"], f" {r['method']}", va="center")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Number of gradient updates")
    ax.set_title("(a)")

    # (b) IM1 vs IM2 with 95% CI
    ax = axes[1]
    for _, r in summary.iterrows():
        ax.errorbar(
            r["mean_im1"],
            r["mean_im2"],
            xerr=r["ci95_im1"],
            yerr=r["ci95_im2"],
            fmt="o",
            capsize=3,
            label=r["method"],
        )
        ax.text(r["mean_im1"], r["mean_im2"], f" {r['method']}", va="center")
    ax.set_xlabel("IM1")
    ax.set_ylabel("IM2")
    ax.set_title("(b)")

    # (c) EN horizontal bars with 95% CI
    ax = axes[2]
    ordered = summary.set_index("method").loc[METHODS].reset_index()
    y = np.arange(len(ordered))
    ax.barh(y, ordered["mean_en"], xerr=ordered["ci95_en"], capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(ordered["method"])
    ax.invert_yaxis()
    ax.set_xlabel("EN(δ)")
    ax.set_ylabel("Loss Function")
    ax.set_title("(c)")

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("figure3.png"))
    p.add_argument("--summary-csv", type=Path, default=Path("figure3_summary.csv"))
    p.add_argument("--summary-json", type=Path, default=Path("figure3_summary.json"))
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--weights-dir", type=Path, default=Path("weights"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--samples-per-class", type=int, default=6)
    p.add_argument("--num-random-seeds", type=int, default=1)
    p.add_argument("--max-updates", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--clip-min", type=float, default=0.0)
    p.add_argument("--clip-max", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device))

    set_determinism(args.seed)

    f_pred, ae_all, ae_class, encoder = load_project_components(device, args.weights_dir)

    test_ds = datasets.MNIST(root="./", train=False, download=True, transform=transforms.ToTensor())
    train_ds = datasets.MNIST(root="./", train=True, download=True, transform=transforms.ToTensor())

    train_imgs = torch.stack([x for x, _ in train_ds], dim=0).to(device)
    train_labels = torch.tensor([y for _, y in train_ds], device=device)
    class_train = {c: train_imgs[train_labels == c] for c in range(10)}

    existing_gen = maybe_get_existing_generator(f_pred=f_pred, ae_all=ae_all, ae_class=ae_class, encoder=encoder, class_train=class_train, device=device)
    generator = existing_gen or FallbackCFGenerator(f_pred, ae_all, encoder, class_train, device)

    labels_test = torch.tensor([y for _, y in test_ds])
    rng = np.random.default_rng(args.seed)
    sampled_indices = sample_balanced_test_indices(labels_test, args.samples_per_class, rng)

    seed_list = [args.seed + i for i in range(args.num_random_seeds)]

    hparams = {
        "c": 1.0,
        "kappa": 0.0,
        "beta": 0.1,
        "gamma": 100.0,
        "K": 15,
    }

    records = []
    total = len(sampled_indices) * len(seed_list) * len(METHODS)
    pbar = tqdm(total=total, desc="Running Figure 3 protocol")

    for idx in sampled_indices:
        x0, y0 = test_ds[idx]
        x0 = x0.unsqueeze(0).to(device)
        true_class = int(y0)

        for s in seed_list:
            for method in METHODS:
                theta = 200.0 if method in {"C", "E"} else (100.0 if method in {"D", "F"} else 0.0)
                run = generator.generate(
                    x0=x0,
                    method=method,
                    seed=s,
                    c=hparams["c"],
                    kappa=hparams["kappa"],
                    beta=hparams["beta"],
                    gamma=hparams["gamma"],
                    theta=theta,
                    K=hparams["K"],
                    max_updates=args.max_updates,
                    lr=args.lr,
                    clip_min=args.clip_min,
                    clip_max=args.clip_max,
                )

                en = hparams["beta"] * torch.norm(run.delta, p=1).item() + torch.norm(run.delta, p=2).pow(2).item()
                im1, im2 = compute_im1_im2(run.x_cf, run.original_class, run.counterfactual_class, ae_class, ae_all)

                records.append(
                    {
                        "method": method,
                        "seed": s,
                        "sample_index": int(idx),
                        "true_class": true_class,
                        "original_class": run.original_class,
                        "counterfactual_class": run.counterfactual_class,
                        "time_seconds": run.time_seconds,
                        "num_grad_updates_used": run.num_grad_updates_used,
                        "EN": en,
                        "IM1": im1,
                        "IM2": im2,
                        "is_satisfactory": int(run.counterfactual_class != run.original_class),
                    }
                )
                pbar.update(1)

    pbar.close()

    raw_df = pd.DataFrame(records)
    summary = aggregate(raw_df)

    make_figure(summary, args.out)

    raw_csv = args.summary_csv.with_name(args.summary_csv.stem + "_raw.csv")
    raw_df.to_csv(raw_csv, index=False)
    summary.to_csv(args.summary_csv, index=False)

    payload = {
        "config": {
            "seed": args.seed,
            "samples_per_class": args.samples_per_class,
            "num_random_seeds": args.num_random_seeds,
            "methods": METHODS,
            **hparams,
            "max_updates": args.max_updates,
            "lr": args.lr,
            "clip_min": args.clip_min,
            "clip_max": args.clip_max,
            "device": str(device),
        },
        "summary": summary.to_dict(orient="records"),
    }
    args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved figure: {args.out}")
    print(f"Saved raw records: {raw_csv}")
    print(f"Saved summary CSV: {args.summary_csv}")
    print(f"Saved summary JSON: {args.summary_json}")


if __name__ == "__main__":
    main()
