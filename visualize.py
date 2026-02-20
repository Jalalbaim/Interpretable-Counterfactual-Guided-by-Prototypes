import argparse
import os
import random
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from algorithm import Counterfactuals
from models.Autoencoder import AutoEncoder
from models.Model_MNIST import Model


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
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

    return model, autoencoder, autoencoder.encoder


def build_reducer(random_state=42):
    try:
        import umap

        return "umap", umap.UMAP(n_components=2, random_state=random_state)
    except ImportError:
        from sklearn.manifold import TSNE

        return "tsne", TSNE(n_components=2, random_state=random_state, init="pca", learning_rate="auto")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize latent space with prototypes and counterfactuals")
    parser.add_argument("--index", type=int, default=None, help="MNIST test index. Random if omitted.")
    parser.add_argument("--method", type=str, default="none", choices=["none", "kmeans"], help="Prototype method")
    parser.add_argument("--k", type=int, default=1, help="K for nearest/kmeans prototypes")
    parser.add_argument("--subset", type=int, default=4000, help="Number of train samples for embedding")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    method = None if args.method == "none" else args.method

    os.makedirs("./outputs", exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root="./", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root="./", train=False, download=True, transform=transform)

    model = Model()
    autoencoder = AutoEncoder()
    model, autoencoder, encoder = trained_model(
        model,
        autoencoder,
        "./weights/mnist_cgp_model_weights.pth",
        "./weights/autoencoder_mnist.pth",
        device,
    )

    model.eval()
    encoder.eval()

    index = args.index if args.index is not None else random.randint(0, len(test_dataset) - 1)
    x_orig = test_dataset[index][0].unsqueeze(0).to(device)

    subset = min(args.subset, len(train_dataset))
    train_data = torch.stack([train_dataset[i][0] for i in range(subset)], dim=0).to(device)

    cf_gen = Counterfactuals(model, encoder, autoencoder, device=device)
    x_counterfactual, details = cf_gen.algorithm_CGP(
        x_orig,
        train_data,
        c=1.0,
        beta=0.1,
        theta=200,
        cap=0.01,
        gamma=100,
        K=args.k,
        max_iterations=500,
        lr=1e-2,
        proto_method=method,
        device=device,
        writer=None,
        return_details=True,
    )

    # predicted labels for training subset
    with torch.no_grad():
        batch_size = 128
        preds = []
        for i in range(0, train_data.size(0), batch_size):
            logits = model(train_data[i:i + batch_size])
            preds.append(logits.argmax(dim=1))
        predicted_labels = torch.cat(preds).cpu().numpy()

        train_enc = encoder(train_data).flatten(1)
        x_orig_enc = encoder(x_orig).flatten(1)
        x_cf_enc = encoder(x_counterfactual).flatten(1)


    predicted_labels_t = torch.from_numpy(predicted_labels)
    class_samples = {cls: train_data[predicted_labels_t == cls] for cls in np.unique(predicted_labels)}
    # Use compute_all_prototypes to get ALL K prototypes per class
    all_prototypes = cf_gen.compute_all_prototypes(x_orig, class_samples, K=args.k, method=method)

    proto_vectors = []
    proto_classes = []
    for cls, proto_list in all_prototypes.items():
        for proto in proto_list:
            proto_vectors.append(proto.flatten().detach().cpu().numpy())
            proto_classes.append(cls)

    all_vectors = np.concatenate(
        [
            train_enc.detach().cpu().numpy(),
            np.stack(proto_vectors),
            x_orig_enc.detach().cpu().numpy(),
            x_cf_enc.detach().cpu().numpy(),
        ],
        axis=0,
    )

    reducer_name, reducer = build_reducer(random_state=42)
    projected = reducer.fit_transform(all_vectors)

    n_train = train_enc.shape[0]
    n_proto = len(proto_vectors)

    train_proj = projected[:n_train]
    proto_proj = projected[n_train:n_train + n_proto]
    orig_proj = projected[n_train + n_proto]
    cf_proj = projected[n_train + n_proto + 1]

    
    with torch.no_grad():
        pred_cf = model(x_counterfactual).argmax(dim=1).item()
        pred_org = model(x_orig).argmax(dim=1).item()

    # Saves

    # save plot side by side of x_orig and x_counterfactual
    fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
    axes[0].imshow(x_orig.detach().cpu().numpy()[0][0], cmap='gray')
    axes[0].set_title(f'Original - Class {pred_org}')
    axes[0].axis('off')

    axes[1].imshow(x_counterfactual.detach().cpu().numpy()[0][0], cmap='gray')
    axes[1].set_title(f'Counterfactual - Class {pred_cf}')
    axes[1].axis('off')
    
    method_label = "nearest" if method is None else method
    image_path = f'./outputs/{ts}_side_{method_label}_{pred_org}_to_{pred_cf}.png'
    print(f"Saved side-by-side comparison to {image_path}")
    plt.savefig(image_path)
    plt.close()

    plt.figure(figsize=(10, 8))
    plt.scatter(train_proj[:, 0], train_proj[:, 1], c=predicted_labels, cmap="tab10", alpha=0.4, s=10)
    plt.scatter(proto_proj[:, 0], proto_proj[:, 1], c=proto_classes, cmap="tab10", marker="d", s=100, edgecolors="black", linewidths=0.5, label="Class prototypes")
    plt.scatter(orig_proj[0], orig_proj[1], c="black", marker="o", s=140, label="x_orig")
    plt.scatter(cf_proj[0], cf_proj[1], c="red", marker="*", s=240, label="x_counterfactual")
    plt.annotate("", xy=(cf_proj[0], cf_proj[1]), xytext=(orig_proj[0], orig_proj[1]), arrowprops=dict(arrowstyle="-", lw=1, color="darkred"))

    plt.title(f"Latent Space ({reducer_name.upper()}) method={method} K={args.k} idx={index}")
    plt.set_cmap("tab10")
    plt.colorbar()
    plt.legend(loc="best")
    plt.tight_layout()

    out_path = f"./outputs/{ts}_latent_space_{method_label}_K{args.k}_idx{index}.png"
    plt.savefig(out_path, dpi=180)
    plt.close()

    print(f"Saved latent space plot to {out_path}")
    print(f"Original class: {details['orig_class']}, target class: {details['target_class']}, cf class: {details['cf_class']}")


if __name__ == "__main__":
    main()

# python .\visualize.py --index 3  --method kmeans --k 3 --subset 4000