import os

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from dataset import BlenderNeRFDataset
from encoding import PositionalEncoding
from model import NeRF
from rays import get_rays_from_pixels
from render import render_rays


def save_checkpoint(path, model, optimizer, step, loss, psnr):
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "psnr": psnr,
    }, path)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs("checkpoints", exist_ok=True)

    # Load one image from Lego
    ds = BlenderNeRFDataset("data/lego", split="train")
    sample = ds[0]

    image = sample["image"].to(device)   # [H, W, 3]
    pose = sample["pose"].to(device)     # [4, 4]
    H, W, focal = sample["H"], sample["W"], sample["focal"]

    # Encoders
    pos_encoder = PositionalEncoding(input_dims=3, num_freqs=10).to(device)
    dir_encoder = PositionalEncoding(input_dims=3, num_freqs=4).to(device)

    # Model
    model = NeRF(
        pos_in_dims=pos_encoder.out_dim,
        dir_in_dims=dir_encoder.out_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    # Render settings
    near = 2.0
    far = 6.0
    n_samples = 32

    # Training settings
    n_iters = 5000
    batch_size = 1024

    model.train()
    for step in range(n_iters):
        # Random pixel coordinates
        i = torch.randint(0, W, (batch_size,), device=device)
        j = torch.randint(0, H, (batch_size,), device=device)

        # Ground truth RGB
        target_rgb = image[j, i]   # [batch_size, 3]

        # Generate rays for those pixels
        rays_o, rays_d = get_rays_from_pixels(i, j, H, W, focal, pose)

        # Render predicted RGB
        pred_rgb, depth_map, acc_map = render_rays(
            model=model,
            pos_encoder=pos_encoder,
            dir_encoder=dir_encoder,
            rays_o=rays_o,
            rays_d=rays_d,
            near=near,
            far=far,
            n_samples=n_samples,
            perturb=True,
            white_background=True,
        )

        loss = F.mse_loss(pred_rgb, target_rgb)
        psnr = -10.0 * torch.log10(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.6f} | PSNR: {psnr.item():.2f}")

        if step % 500 == 0 or step == n_iters - 1:
            save_checkpoint(
                path=f"checkpoints/nerf_one_image_step_{step}.pt",
                model=model,
                optimizer=optimizer,
                step=step,
                loss=loss.item(),
                psnr=psnr.item(),
            )

    # Full-image render after training
    model.eval()
    with torch.no_grad():
        pred_image = torch.zeros((H, W, 3), device=device)

        chunk = 4096
        all_i, all_j = torch.meshgrid(
            torch.arange(W, device=device),
            torch.arange(H, device=device),
            indexing="xy"
        )
        all_i = all_i.reshape(-1)
        all_j = all_j.reshape(-1)

        for start in range(0, all_i.shape[0], chunk):
            end = start + chunk
            i_chunk = all_i[start:end]
            j_chunk = all_j[start:end]

            rays_o, rays_d = get_rays_from_pixels(i_chunk, j_chunk, H, W, focal, pose)

            pred_rgb, _, _ = render_rays(
                model=model,
                pos_encoder=pos_encoder,
                dir_encoder=dir_encoder,
                rays_o=rays_o,
                rays_d=rays_d,
                near=near,
                far=far,
                n_samples=n_samples,
                perturb=False,
                white_background=True,
            )

            pred_image[j_chunk, i_chunk] = pred_rgb

        pred_image = pred_image.clamp(0.0, 1.0).cpu().numpy()
        gt_image = image.cpu().numpy()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(gt_image)
    plt.title("Ground Truth")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_image)
    plt.title("NeRF Overfit Result")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()