"""Script pour exécuter uniquement make_qualitative_figure"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict

import torch
import numpy as np
from torchvision import datasets, transforms

from experiment import (
    CFRunResult,
    FallbackCFGenerator,
    load_project_components,
    make_qualitative_figure,
    set_determinism,
    METHODS,
    maybe_get_existing_generator,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Générer uniquement la figure qualitative")
    p.add_argument("--out", type=Path, default=Path("qualitative_figure.png"), 
                   help="Chemin de sortie pour la figure")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="Device à utiliser")
    p.add_argument("--weights-dir", type=Path, default=Path("weights"),
                   help="Répertoire contenant les poids des modèles")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed pour la reproductibilité")
    p.add_argument("--test-index", type=int, default=0,
                   help="Index de l'exemple du jeu de test à utiliser")
    p.add_argument("--max-updates", type=int, default=500,
                   help="Nombre maximum de mises à jour du gradient")
    p.add_argument("--lr", type=float, default=1e-2,
                   help="Learning rate pour l'optimisation")
    p.add_argument("--clip-min", type=float, default=0.0,
                   help="Valeur minimale pour le clipping")
    p.add_argument("--clip-max", type=float, default=1.0,
                   help="Valeur maximale pour le clipping")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    
    # Configuration du device
    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) 
        else ("cpu" if args.device == "auto" else args.device)
    )
    print(f"Utilisation du device: {device}")
    
    # Déterminisme
    set_determinism(args.seed)
    
    # Chargement des composants du projet
    print("Chargement des modèles...")
    f_pred, ae_all, ae_class, encoder = load_project_components(device, args.weights_dir)
    
    # Chargement des données
    print("Chargement des données MNIST...")
    test_ds = datasets.MNIST(root="./", train=False, download=True, 
                            transform=transforms.ToTensor())
    train_ds = datasets.MNIST(root="./", train=True, download=True, 
                             transform=transforms.ToTensor())
    
    # Préparation des données d'entraînement par classe
    train_imgs = torch.stack([x for x, _ in train_ds], dim=0).to(device)
    train_labels = torch.tensor([y for _, y in train_ds], device=device)
    class_train = {c: train_imgs[train_labels == c] for c in range(10)}
    
    # Création du générateur
    existing_gen = maybe_get_existing_generator(
        f_pred=f_pred, ae_all=ae_all, ae_class=ae_class, 
        encoder=encoder, class_train=class_train, device=device
    )
    generator = existing_gen or FallbackCFGenerator(f_pred, ae_all, encoder, class_train, device)
    
    # Sélection de l'exemple de test
    if not (0 <= args.test_index < len(test_ds)):
        raise ValueError(f"test-index doit être dans [0, {len(test_ds)-1}], reçu {args.test_index}")
    
    x0, y0 = test_ds[args.test_index]
    x0 = x0.unsqueeze(0).to(device)
    print(f"Exemple sélectionné: index {args.test_index}, classe réelle: {y0}")
    
    # Prédiction de la classe d'origine
    with torch.no_grad():
        pred_class = int(torch.argmax(f_pred(x0), dim=1).item())
    print(f"Classe prédite par le modèle: {pred_class}")
    
    # Hyperparamètres
    hparams = {
        "c": 1.0,
        "kappa": 0.0,
        "beta": 0.1,
        "gamma": 100.0,
        "K": 15,
    }
    
    # Génération des counterfactuals pour chaque méthode
    print(f"\nGénération des counterfactuals pour les méthodes {METHODS}...")
    qual_runs: Dict[str, CFRunResult] = {}
    
    for method in METHODS:
        print(f"  Méthode {method}...", end=" ")
        theta = 200.0 if method in {"C", "E"} else (100.0 if method in {"D", "F"} else 0.0)
        
        run = generator.generate(
            x0=x0,
            method=method,
            seed=args.seed,
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
        
        qual_runs[method] = run
        print(f"Classe CF: {run.counterfactual_class}, "
              f"Étapes: {run.num_grad_updates_used}, "
              f"Temps: {run.time_seconds:.3f}s")
    
    # Génération de la figure qualitative
    print(f"\nCréation de la figure qualitative...")
    make_qualitative_figure(x0, qual_runs, ae_all, args.out)
    
    print(f"✓ Figure sauvegardée: {args.out}")


if __name__ == "__main__":
    main()
