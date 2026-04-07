import os
import csv
import math
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from dataset import BlenderNeRFDataset
from encoding import PositionalEncoding
from model import NeRF
from rays import get_rays_from_pixels
from render import render_rays_hierarchical, volume_render, sample_points_on_rays


def save_checkpoint(path, coarse_model, fine_model, optimizer, step, loss, psnr):
    torch.save({
        "step": step,
        "coarse_state_dict":    coarse_model.state_dict(),
        "fine_state_dict":      fine_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "psnr": psnr,
    }, path)


def render_full_image_hierarchical(
    coarse_model,
    fine_model,
    pos_encoder,
    dir_encoder,
    pose,
    H,
    W,
    focal,
    near,
    far,
    n_coarse,
    n_fine,
    device,
    chunk=4096,
):
    coarse_model.eval()
    fine_model.eval()
    pred_image = torch.zeros((H, W, 3), device=device)

    with torch.no_grad():
        all_i, all_j = torch.meshgrid(
            torch.arange(W, device=device),
            torch.arange(H, device=device),
            indexing="xy",
        )
        all_i = all_i.reshape(-1)
        all_j = all_j.reshape(-1)

        for start in range(0, all_i.shape[0], chunk):
            end      = min(start + chunk, all_i.shape[0])
            i_chunk  = all_i[start:end]
            j_chunk  = all_j[start:end]

            rays_o, rays_d = get_rays_from_pixels(i_chunk, j_chunk, H, W, focal, pose)

            _, (pred_rgb, _, _) = render_rays_hierarchical(
                coarse_model=coarse_model,
                fine_model=fine_model,
                pos_encoder=pos_encoder,
                dir_encoder=dir_encoder,
                rays_o=rays_o,
                rays_d=rays_d,
                near=near,
                far=far,
                n_coarse=n_coarse,
                n_fine=n_fine,
                perturb=False,
                white_background=True,
            )

            pred_image[j_chunk, i_chunk] = pred_rgb

    coarse_model.train()
    fine_model.train()
    return pred_image.clamp(0.0, 1.0)


def render_dataset_views(
    coarse_model,
    fine_model,
    pos_encoder,
    dir_encoder,
    dataset,
    near,
    far,
    n_coarse,
    n_fine,
    device,
    num_views=4,
    save_path=None,
):
    H, W, focal = dataset[0]["H"], dataset[0]["W"], dataset[0]["focal"]
    fig, axes = plt.subplots(num_views, 2, figsize=(8, 4 * num_views))

    for idx in range(num_views):
        sample   = dataset[idx]
        gt_image = sample["image"].to(device)
        pose     = sample["pose"].to(device)

        pred_image = render_full_image_hierarchical(
            coarse_model=coarse_model,
            fine_model=fine_model,
            pos_encoder=pos_encoder,
            dir_encoder=dir_encoder,
            pose=pose,
            H=H, W=W, focal=focal,
            near=near, far=far,
            n_coarse=n_coarse, n_fine=n_fine,
            device=device,
        )

        axes[idx, 0].imshow(gt_image.cpu().numpy())
        axes[idx, 0].set_title(f"GT view {idx}")
        axes[idx, 0].axis("off")

        axes[idx, 1].imshow(pred_image.cpu().numpy())
        axes[idx, 1].set_title(f"Rendered view {idx}")
        axes[idx, 1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved dataset views → {save_path}")
    plt.show()


def trans_t(t):
    return torch.tensor([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, t],
        [0, 0, 0, 1],
    ], dtype=torch.float32)


def rot_phi(phi):
    return torch.tensor([
        [1,           0,            0, 0],
        [0, math.cos(phi), -math.sin(phi), 0],
        [0, math.sin(phi),  math.cos(phi), 0],
        [0,           0,            0, 1],
    ], dtype=torch.float32)


def rot_theta(theta):
    return torch.tensor([
        [ math.cos(theta), 0, math.sin(theta), 0],
        [               0, 1,               0, 0],
        [-math.sin(theta), 0, math.cos(theta), 0],
        [               0, 0,               0, 1],
    ], dtype=torch.float32)


def pose_spherical(theta_deg, phi_deg, radius):
    theta = math.radians(theta_deg)
    phi   = math.radians(phi_deg)

    c2w = trans_t(radius)
    c2w = rot_phi(phi) @ c2w
    c2w = rot_theta(theta) @ c2w

    convert = torch.tensor([
        [-1, 0, 0, 0],
        [ 0, 0, 1, 0],
        [ 0, 1, 0, 0],
        [ 0, 0, 0, 1],
    ], dtype=torch.float32)

    return convert @ c2w


def render_orbit_frames(
    coarse_model,
    fine_model,
    pos_encoder,
    dir_encoder,
    H,
    W,
    focal,
    near,
    far,
    n_coarse,
    n_fine,
    device,
    n_frames=60,
    radius=4.0,
    phi_deg=-30.0,
):
    frames = []
    for k in range(n_frames):
        theta_deg = 360.0 * k / n_frames
        pose      = pose_spherical(theta_deg, phi_deg, radius).to(device)

        pred_image = render_full_image_hierarchical(
            coarse_model=coarse_model,
            fine_model=fine_model,
            pos_encoder=pos_encoder,
            dir_encoder=dir_encoder,
            pose=pose,
            H=H, W=W, focal=focal,
            near=near, far=far,
            n_coarse=n_coarse, n_fine=n_fine,
            device=device,
        )

        frame = (pred_image.cpu().numpy() * 255).astype(np.uint8)
        frames.append(frame)
        print(f"Rendered orbit frame {k + 1}/{n_frames}")

    return frames


def save_gif(frames, path, fps=20):
    imageio.mimsave(path, frames, fps=fps)
    print(f"Saved GIF → {path}")


def plot_training_curves(history, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["step"], history["loss"], color="tab:orange", linewidth=1.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("MSE loss (log scale)")
    axes[0].set_title("Training loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["step"], history["psnr"], color="tab:green", linewidth=1.5)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Training PSNR")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved training curves → {save_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs",     exist_ok=True)

    # ── datasets ─────────────────────────────────────────────────
    train_ds = BlenderNeRFDataset("data/lego", split="train")
    test_ds  = BlenderNeRFDataset("data/lego", split="test")

    H, W, focal = train_ds[0]["H"], train_ds[0]["W"], train_ds[0]["focal"]

    # ── encoders ─────────────────────────────────────────────────
    pos_encoder = PositionalEncoding(input_dims=3, num_freqs=12).to(device)
    dir_encoder = PositionalEncoding(input_dims=3, num_freqs=4).to(device)

    # ── models ───────────────────────────────────────────────────
    coarse_model = NeRF(
        pos_in_dims=pos_encoder.out_dim,
        dir_in_dims=dir_encoder.out_dim,
    ).to(device)

    fine_model = NeRF(
        pos_in_dims=pos_encoder.out_dim,
        dir_in_dims=dir_encoder.out_dim,
    ).to(device)

    optimizer = torch.optim.Adam(
        list(coarse_model.parameters()) + list(fine_model.parameters()),
        lr=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

    # ── render settings ──────────────────────────────────────────
    near     = 2.0
    far      = 6.0
    n_coarse = 64
    n_fine   = 128

    # ── training settings ────────────────────────────────────────
    n_iters    = 10000
    batch_size = 4096

    # ── logging setup ────────────────────────────────────────────
    log_path = Path("outputs/training_log.csv")
    history  = {"step": [], "loss": [], "psnr": [], "acc_mean": []}

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss", "psnr", "acc_mean"])

    # ── training loop ────────────────────────────────────────────
    for step in range(n_iters):
        img_idx = torch.randint(0, len(train_ds), (1,)).item()
        sample  = train_ds[img_idx]

        image = sample["image"].to(device)
        pose  = sample["pose"].to(device)

        i = torch.randint(0, W, (batch_size,), device=device)
        j = torch.randint(0, H, (batch_size,), device=device)

        target_rgb = image[j, i]
        rays_o, rays_d = get_rays_from_pixels(i, j, H, W, focal, pose)

        (pred_rgb_c, _, acc_map), (pred_rgb_f, _, _) = render_rays_hierarchical(
            coarse_model=coarse_model,
            fine_model=fine_model,
            pos_encoder=pos_encoder,
            dir_encoder=dir_encoder,
            rays_o=rays_o,
            rays_d=rays_d,
            near=near,
            far=far,
            n_coarse=n_coarse,
            n_fine=n_fine,
            perturb=True,
            white_background=True,
        )

        # Eq. 6 — loss on both coarse and fine outputs
        loss_c = F.mse_loss(pred_rgb_c, target_rgb)
        loss_f = F.mse_loss(pred_rgb_f, target_rgb)
        loss   = loss_c + loss_f
        psnr   = -10.0 * torch.log10(loss_f)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            loss_val = loss_f.item()
            psnr_val = psnr.item()
            acc_val  = acc_map.mean().item()

            print(
                f"Step {step:6d} | "
                f"Loss (fine): {loss_val:.6f} | "
                f"PSNR: {psnr_val:.2f} dB | "
                f"Acc mean: {acc_val:.4f}"
            )

            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, loss_val, psnr_val, acc_val])

            history["step"].append(step)
            history["loss"].append(loss_val)
            history["psnr"].append(psnr_val)
            history["acc_mean"].append(acc_val)

        if step % 1000 == 0 or step == n_iters - 1:
            save_checkpoint(
                path=f"checkpoints/nerf_lego_step_{step}.pt",
                coarse_model=coarse_model,
                fine_model=fine_model,
                optimizer=optimizer,
                step=step,
                loss=loss.item(),
                psnr=psnr.item(),
            )

    # ── training curves ──────────────────────────────────────────
    plot_training_curves(history, save_path="outputs/training_curves.png")

    # ── test evaluation (single view) ────────────────────────────
    test_sample = test_ds[0]
    test_image  = test_sample["image"].to(device)
    test_pose   = test_sample["pose"].to(device)

    pred_image = render_full_image_hierarchical(
        coarse_model=coarse_model,
        fine_model=fine_model,
        pos_encoder=pos_encoder,
        dir_encoder=dir_encoder,
        pose=test_pose,
        H=H, W=W, focal=focal,
        near=near, far=far,
        n_coarse=n_coarse, n_fine=n_fine,
        device=device,
    )

    pred_np = pred_image.cpu().numpy()
    gt_np   = test_image.cpu().numpy()

    test_loss = F.mse_loss(pred_image, test_image).item()
    test_psnr = -10.0 * math.log10(test_loss)
    print(f"\nTest Loss: {test_loss:.6f}")
    print(f"Test PSNR: {test_psnr:.2f} dB")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(gt_np);   axes[0].set_title("Ground truth"); axes[0].axis("off")
    axes[1].imshow(pred_np); axes[1].set_title("NeRF render");  axes[1].axis("off")
    plt.tight_layout()
    plt.savefig("outputs/test_view_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ── multi-view comparison ─────────────────────────────────────
    render_dataset_views(
        coarse_model=coarse_model,
        fine_model=fine_model,
        pos_encoder=pos_encoder,
        dir_encoder=dir_encoder,
        dataset=test_ds,
        near=near, far=far,
        n_coarse=n_coarse, n_fine=n_fine,
        device=device,
        num_views=4,
        save_path="outputs/dataset_views.png",
    )

    # ── orbit animation ───────────────────────────────────────────
    orbit_frames = render_orbit_frames(
        coarse_model=coarse_model,
        fine_model=fine_model,
        pos_encoder=pos_encoder,
        dir_encoder=dir_encoder,
        H=H, W=W, focal=focal,
        near=near, far=far,
        n_coarse=n_coarse, n_fine=n_fine,
        device=device,
        n_frames=60,
        radius=4.0,
        phi_deg=-30.0,
    )
    save_gif(orbit_frames, "outputs/lego_orbit.gif", fps=20)


if __name__ == "__main__":
    main()