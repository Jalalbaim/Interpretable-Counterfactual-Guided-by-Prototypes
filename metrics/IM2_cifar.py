"""IM2 interpretability metric for CIFAR-10 counterfactuals."""

import torch


_DEF_SHAPE = (3, 32, 32)


def _validate_x_cf(x_cf: torch.Tensor) -> None:
    if not isinstance(x_cf, torch.Tensor):
        raise TypeError(f"x_cf must be a torch.Tensor, got {type(x_cf)}")
    if x_cf.ndim != 4:
        raise ValueError(f"x_cf must have shape (N, 3, 32, 32), got tensor with ndim={x_cf.ndim}")
    if tuple(x_cf.shape[1:]) != _DEF_SHAPE:
        raise ValueError(f"x_cf must have shape (N, 3, 32, 32), got {tuple(x_cf.shape)}")


def _validate_reduction(reduction: str) -> None:
    if reduction not in {"none", "mean"}:
        raise ValueError(f"Unsupported reduction='{reduction}'. Expected 'none' or 'mean'.")


def compute_im2(x_cf, ae_i, ae_all, eps=1e-8, reduction="none"):
    """
    IM2 = || AE_i(x_cf) - AE_all(x_cf) ||_2^2 / ( || x_cf ||_1 + eps )

    Returns:
        torch.Tensor of shape (N,) if reduction='none', otherwise scalar tensor.
    """
    _validate_x_cf(x_cf)
    _validate_reduction(reduction)

    ae_i.eval()
    ae_all.eval()

    with torch.no_grad():
        x_cf = x_cf.to(x_cf.device)
        recon_i = ae_i(x_cf)
        recon_all = ae_all(x_cf)

        if recon_i.shape != x_cf.shape:
            raise ValueError(f"AE_i output shape must match x_cf. Got {tuple(recon_i.shape)} vs {tuple(x_cf.shape)}")
        if recon_all.shape != x_cf.shape:
            raise ValueError(f"AE_all output shape must match x_cf. Got {tuple(recon_all.shape)} vs {tuple(x_cf.shape)}")

        num = (recon_i - recon_all).flatten(1).pow(2).sum(dim=1)
        den = x_cf.flatten(1).abs().sum(dim=1) + eps
        im2 = num / den

    return im2.mean() if reduction == "mean" else im2
