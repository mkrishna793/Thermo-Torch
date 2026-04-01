"""
Latent PFE Example

Demonstrates training PFE in a VAE latent space for high-dimensional data.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Note: This requires diffusers for the VAE
# pip install diffusers transformers

def latent_pfe_example():
    """
    Complete example of training Latent PFE on image data.

    Pipeline:
        1. Load images
        2. Encode to latent space via frozen VAE
        3. Train PFE in latent space
        4. Settle and decode for generation
    """
    print("Latent PFE Example")
    print("=" * 50)

    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    latent_dim = 4 * 64 * 64  # Standard SD VAE latent: 4 channels x 64x64

    print(f"Device: {device}")
    print(f"Latent dimension: {latent_dim}")

    # Step 1: Load VAE (frozen)
    print("\n[1] Loading VAE encoder...")
    try:
        from diffusers import AutoencoderKL
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        vae = vae.to(device)
        vae.eval()
        for param in vae.parameters():
            param.requires_grad = False
        print("    VAE loaded successfully!")
    except ImportError:
        print("    VAE requires diffusers. Install with: pip install diffusers")
        print("    Skipping VAE example...")
        return

    # Step 2: Create PFE model for latent space
    print("\n[2] Creating LatentPFEEncoder...")
    from thermotorch.core import PFEEncoder, LatentPFEEncoder

    pfe_encoder = PFEEncoder(embed_dim=latent_dim, hidden_dim=latent_dim * 2)
    latent_pfe = LatentPFEEncoder(vae, pfe_encoder).to(device)
    print("    Latent PFE model created!")

    # Step 3: Create loss and optimizer
    print("\n[3] Setting up training...")
    from thermotorch.core import PFELoss

    criterion = PFELoss(sigma=0.1, alpha=0.1, beta=0.5)
    optimizer = torch.optim.AdamW(pfe_encoder.parameters(), lr=1e-4)
    print("    Loss and optimizer ready!")

    # Step 4: Load image data
    print("\n[4] Loading image dataset...")
    transform = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # Use MNIST as example (replace with your dataset)
    dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"    Dataset: {len(dataset)} samples")

    # Step 5: Training loop (short demo)
    print("\n[5] Training (10 iterations demo)...")
    pfe_encoder.train()

    for i, (images, _) in enumerate(dataloader):
        if i >= 10:
            break

        images = images.to(device)

        # Create positive pairs (augmented)
        images_pos = images + torch.randn_like(images) * 0.05

        # Create noise samples
        noise = torch.randn_like(images)

        # Forward pass
        optimizer.zero_grad()
        z, s, v_raw, v_unit = latent_pfe(images)
        z_pos, s_pos, v_raw_pos, v_unit_pos = latent_pfe(images_pos)
        z_noise, s_noise, v_raw_noise, v_unit_noise = latent_pfe(noise)

        # Compute loss using the PFE encoder directly on latents
        loss, l_dsm, l_nce, l_infonce = criterion(
            pfe_encoder, z, z_pos, z_noise, return_components=True
        )

        # Backward
        loss.backward()
        optimizer.step()

        if i % 5 == 0:
            print(f"    Iter {i}: Loss={loss.item():.4f}")

    print("\n[6] Generation demo...")
    print("    Generating from random noise...")

    pfe_encoder.eval()
    with torch.no_grad():
        # Random noise in latent space
        noisy_latent = torch.randn(4, latent_dim).to(device)

        # Settle to attractor
        from thermotorch.core import tsu_settle
        settled_latent = tsu_settle(pfe_encoder, noisy_latent, steps=50)

        # Decode to image
        # Reshape for VAE decoder
        settled_z = settled_latent.view(4, 4, 64, 64)
        generated = vae.decode(settled_z).sample

        print(f"    Generated images shape: {generated.shape}")
        print("    (Images would be saved here)")

    print("\n" + "=" * 50)
    print("Latent PFE example complete!")


if __name__ == "__main__":
    latent_pfe_example()