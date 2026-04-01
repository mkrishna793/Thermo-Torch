"""
PFE Training Example

Complete training pipeline for Potential-Flow Embeddings.
This example demonstrates:
    1. Creating a PFEEncoder model
    2. Setting up the tri-partite loss (DSM + NCE + InfoNCE)
    3. Training loop with validation
    4. TSU settling for inference

Usage:
    python train_pfe.py --embed_dim 512 --epochs 100 --batch_size 64
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
import argparse

# Import ThermoTorch components
import sys
sys.path.insert(0, '..')

from thermotorch.core import PFEEncoder, PFELoss, tsu_settle
from thermotorch.core.tsu_settle import SettlingMethod


def create_dummy_dataset(
    num_samples: int = 10000,
    embed_dim: int = 512,
    num_clusters: int = 10
):
    """
    Create a dummy dataset with cluster structure.

    Real applications would use actual embeddings from:
        - Image encoders (ViT, ResNet)
        - Text encoders (BERT, GPT)
        - Multimodal encoders (CLIP)
    """
    # Create cluster centers
    centers = torch.randn(num_clusters, embed_dim) * 3

    # Generate samples around cluster centers
    samples = []
    labels = []
    for _ in range(num_samples):
        cluster_idx = torch.randint(0, num_clusters, (1,)).item()
        noise = torch.randn(embed_dim) * 0.5
        sample = centers[cluster_idx] + noise
        samples.append(sample)
        labels.append(cluster_idx)

    X = torch.stack(samples)
    y = torch.tensor(labels)
    return X, y


def augment(x: torch.Tensor, noise_scale: float = 0.1) -> torch.Tensor:
    """Create positive pairs via simple augmentation."""
    return x + torch.randn_like(x) * noise_scale


def train_pfe(
    embed_dim: int = 512,
    hidden_dim: int = 1024,
    num_layers: int = 2,
    sigma: float = 0.1,
    alpha: float = 0.1,
    beta: float = 0.5,
    temperature: float = 0.07,
    lr: float = 1e-4,
    batch_size: int = 64,
    epochs: int = 100,
    device: str = 'cuda'
):
    """
    Train a PFE model.

    Args:
        embed_dim: Input embedding dimension
        hidden_dim: Hidden layer dimension
        num_layers: Number of backbone layers
        sigma: Noise scale for DSM
        alpha: NCE loss weight
        beta: InfoNCE loss weight
        temperature: InfoNCE temperature
        lr: Learning rate
        batch_size: Training batch size
        epochs: Number of training epochs
        device: Device to train on
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")

    # Create model
    model = PFEEncoder(
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers
    ).to(device)

    # Create loss function
    criterion = PFELoss(
        sigma=sigma,
        alpha=alpha,
        beta=beta,
        temperature=temperature
    )

    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr)

    # Create dataset
    print("Creating dataset...")
    X, y = create_dummy_dataset(num_samples=10000, embed_dim=embed_dim)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        total_dsm = 0.0
        total_nce = 0.0
        total_infonce = 0.0

        for batch_idx, (x, _) in enumerate(dataloader):
            x = x.to(device)

            # Create positive pairs (augmented versions)
            x_pos = augment(x, noise_scale=0.05).to(device)

            # Create noise samples (negative samples)
            x_noise = torch.randn_like(x).to(device)

            # Forward pass
            optimizer.zero_grad()
            loss, l_dsm, l_nce, l_infonce = criterion(
                model, x, x_pos, x_noise, return_components=True
            )

            # Backward pass
            loss.backward()
            optimizer.step()

            # Accumulate losses
            total_loss += loss.item()
            total_dsm += l_dsm.item()
            total_nce += l_nce.item()
            total_infonce += l_infonce.item()

        # Print progress
        n_batches = len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Loss: {total_loss/n_batches:.4f} | "
                f"DSM: {total_dsm/n_batches:.4f} | "
                f"NCE: {total_nce/n_batches:.4f} | "
                f"InfoNCE: {total_infonce/n_batches:.4f}"
            )

    print("Training complete!")

    # Demonstrate TSU settling
    print("\n--- TSU Settling Demo ---")
    model.eval()
    with torch.no_grad():
        # Start with random noise
        noisy_input = torch.randn(4, embed_dim).to(device)

        # Settle to attractor
        settled = tsu_settle(model, noisy_input, steps=50, method=SettlingMethod.EULER)

        # Get energy before and after
        s_before, _, _ = model(noisy_input)
        s_after, _, _ = model(settled)

        print(f"Energy before settling: {s_before.mean().item():.4f}")
        print(f"Energy after settling:  {s_after.mean().item():.4f}")
        print(f"Energy reduction: {(s_before.mean() - s_after.mean()).item():.4f}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Train PFE model")
    parser.add_argument('--embed_dim', type=int, default=512)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--sigma', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    model = train_pfe(
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        sigma=args.sigma,
        alpha=args.alpha,
        beta=args.beta,
        temperature=args.temperature,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        device=args.device
    )

    # Save model
    torch.save(model.state_dict(), 'pfe_model.pt')
    print("Model saved to pfe_model.pt")


if __name__ == "__main__":
    main()