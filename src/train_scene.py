import os
import math
import csv
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
from render import render_rays


def save_checkpoint(path, model, optimizer, step, loss, psnr):
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "psnr": psnr,
    }, path)


def render_full_image(
    model,
    pos_encoder,
    dir_encoder,
    pose,
    H,
    W,
    focal,
    near,
    far,
    n_samples,
    device,
    chunk=4096,
):
    model.eval()
    pred_image = torch.zeros((H, W, 3), device=device)

    with torch.no_grad():
        all_i, all_j = torch.meshgrid(
            torch.arange(W, device=device),
            torch.arange(H, device=device),
            indexing="xy"
        )
        all_i = all_i.reshape(-1)
        all_j = all_j.reshape(-1)

        for start in range(0, all_i.shape[0], chunk):
            end = min(start + chunk, all_i.shape[0])
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

    model.train()
    return pred_image.clamp(0.0, 1.0)


def render_dataset_views(
    model,
    pos_encoder,
    dir_encoder,
    dataset,
    near,
    far,
    n_samples,
    device,
    num_views=4,
):
    H, W, focal = dataset[0]["H"], dataset[0]["W"], dataset[0]["focal"]

    plt.figure(figsize=(10, 4 * num_views))

    for idx in range(num_views):
        sample = dataset[idx]
        gt_image = sample["image"].to(device)
        pose = sample["pose"].to(device)

        pred_image = render_full_image(
            model=model,
            pos_encoder=pos_encoder,
            dir_encoder=dir_encoder,
            pose=pose,
            H=H,
            W=W,
            focal=focal,
            near=near,
            far=far,
            n_samples=n_samples,
            device=device,
            chunk=4096,
        )

        gt_np = gt_image.cpu().numpy()
        pred_np = pred_image.cpu().numpy()

        plt.subplot(num_views, 2, 2 * idx + 1)
        plt.imshow(gt_np)
        plt.title(f"GT View {idx}")
        plt.axis("off")

        plt.subplot(num_views, 2, 2 * idx + 2)
        plt.imshow(pred_np)
        plt.title(f"Rendered View {idx}")
        plt.axis("off")

    plt.tight_layout()
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
        [1, 0, 0, 0],
        [0, math.cos(phi), -math.sin(phi), 0],
        [0, math.sin(phi),  math.cos(phi), 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)


def rot_theta(theta):
    return torch.tensor([
        [ math.cos(theta), 0, -math.sin(theta), 0],
        [0, 1, 0, 0],
        [ math.sin(theta), 0,  math.cos(theta), 0],
        [0, 0, 0, 1],
    ], dtype=torch.float32)


def pose_spherical(theta_deg, phi_deg, radius):
    """
    Standard NeRF orbit camera pose.
    """
    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)

    c2w = trans_t(radius)
    c2w = rot_phi(phi) @ c2w
    c2w = rot_theta(theta) @ c2w

    # Convert to NeRF's coordinate convention
    convert = torch.tensor([
        [-1,  0,  0, 0],
        [ 0,  0,  1, 0],
        [ 0,  1,  0, 0],
        [ 0,  0,  0, 1],
    ], dtype=torch.float32)

    c2w = convert @ c2w
    return c2w


def render_orbit_frames(
    model,
    pos_encoder,
    dir_encoder,
    H,
    W,
    focal,
    near,
    far,
    n_samples,
    device,
    n_frames=60,
    radius=4.0,
    phi_deg=-30.0,
):
    frames = []

    for k in range(n_frames):
        theta_deg = 360.0 * k / n_frames
        pose = pose_spherical(theta_deg, phi_deg, radius).to(device)

        pred_image = render_full_image(
            model=model,
            pos_encoder=pos_encoder,
            dir_encoder=dir_encoder,
            pose=pose,
            H=H,
            W=W,
            focal=focal,
            near=near,
            far=far,
            n_samples=n_samples,
            device=device,
            chunk=4096,
        )

        frame = (pred_image.cpu().numpy() * 255).astype(np.uint8)
        frames.append(frame)

        print(f"Rendered orbit frame {k+1}/{n_frames}")

    return frames


def save_gif(frames, path, fps=20):
    imageio.mimsave(path, frames, fps=fps)
    print(f"Saved GIF to: {path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Datasets
    train_ds = BlenderNeRFDataset("data/lego", split="train")
    test_ds = BlenderNeRFDataset("data/lego", split="test")

    H, W, focal = train_ds[0]["H"], train_ds[0]["W"], train_ds[0]["focal"]

    # Model + encoders
    pos_encoder = PositionalEncoding(input_dims=3, num_freqs=12).to(device)
    dir_encoder = PositionalEncoding(input_dims=3, num_freqs=4).to(device)

    model = NeRF(
        pos_in_dims=pos_encoder.out_dim,
        dir_in_dims=dir_encoder.out_dim
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9999)

    # Render settings
    near = 2.0
    far = 6.0
    n_samples = 128

    # Training settings
    n_iters = 200000
    batch_size = 4096

    # ── logging setup ───────────────────────────────────────────
    log_path = Path("outputs/training_log.csv")
    log_path.parent.mkdir(exist_ok=True)

    history = {"step": [], "loss": [], "psnr": [], "acc_mean": [], "acc_max": []}

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss", "psnr", "acc_mean", "acc_max"])

    # ── training loop ────────────────────────────────────────────
    for step in range(n_iters):
        img_idx = torch.randint(0, len(train_ds), (1,)).item()
        sample = train_ds[img_idx]

        image = sample["image"].to(device)
        pose  = sample["pose"].to(device)

        i = torch.randint(0, W, (batch_size,), device=device)
        j = torch.randint(0, H, (batch_size,), device=device)

        target_rgb = image[j, i]
        rays_o, rays_d = get_rays_from_pixels(i, j, H, W, focal, pose)

        pred_rgb, depth_map, acc_map = render_rays(
            model=model, pos_encoder=pos_encoder, dir_encoder=dir_encoder,
            rays_o=rays_o, rays_d=rays_d, near=near, far=far,
            n_samples=n_samples, perturb=True, white_background=True,
        )

        loss = F.mse_loss(pred_rgb, target_rgb)
        psnr = -10.0 * torch.log10(loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step % 100 == 0:
            loss_val    = loss.item()
            psnr_val    = psnr.item()
            acc_mean    = acc_map.mean().item()
            acc_max     = acc_map.max().item()

            print(
                f"Step {step:6d} | Loss: {loss_val:.6f} | "
                f"PSNR: {psnr_val:.2f} | acc mean: {acc_mean:.4f} | acc max: {acc_max:.4f}"
            )

            # ── append to CSV ────────────────────────────────────
            with open(log_path, "a", newline="") as f:
                csv.writer(f).writerow([step, loss_val, psnr_val, acc_mean, acc_max])

            # ── keep in-memory history ───────────────────────────
            history["step"].append(step)
            history["loss"].append(loss_val)
            history["psnr"].append(psnr_val)
            history["acc_mean"].append(acc_mean)
            history["acc_max"].append(acc_max)

        if step % 1000 == 0 or step == n_iters - 1:
            save_checkpoint(
                path=f"checkpoints/nerf_lego_step_{step}.pt",
                model=model, optimizer=optimizer,
                step=step, loss=loss.item(), psnr=psnr.item(),
            )

    # ── plot training curves ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["step"], history["loss"], color="tab:orange", linewidth=1.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("MSE Loss (log scale)")
    axes[0].set_title("Training loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["step"], history["psnr"], color="tab:green", linewidth=1.5)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("PSNR (dB)")
    axes[1].set_title("Training PSNR")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("outputs/training_curves.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved training curves → outputs/training_curves.png")

    # Render one held-out test view
    test_sample = test_ds[0]
    test_image = test_sample["image"].to(device)
    test_pose = test_sample["pose"].to(device)

    pred_image = render_full_image(
        model=model,
        pos_encoder=pos_encoder,
        dir_encoder=dir_encoder,
        pose=test_pose,
        H=H,
        W=W,
        focal=focal,
        near=near,
        far=far,
        n_samples=n_samples,
        device=device,
        chunk=4096,
    )

    pred_image_np = pred_image.cpu().numpy()
    test_image_np = test_image.cpu().numpy()

    test_loss = F.mse_loss(pred_image, test_image).item()
    test_psnr = -10.0 * torch.log10(torch.tensor(test_loss)).item()

    print(f"\nTest Loss: {test_loss:.6f}")
    print(f"Test PSNR: {test_psnr:.2f}")

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(test_image_np)
    plt.title("Ground Truth Test View")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(pred_image_np)
    plt.title("Rendered Test View")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Render multiple dataset views
    render_dataset_views(
        model=model,
        pos_encoder=pos_encoder,
        dir_encoder=dir_encoder,
        dataset=test_ds,
        near=near,
        far=far,
        n_samples=n_samples,
        device=device,
        num_views=4,
    )

    # Render orbit animation
    orbit_frames = render_orbit_frames(
        model=model,
        pos_encoder=pos_encoder,
        dir_encoder=dir_encoder,
        H=H,
        W=W,
        focal=focal,
        near=near,
        far=far,
        n_samples=n_samples,
        device=device,
        n_frames=60,
        radius=4.0,
        phi_deg=-30.0,
    )

    save_gif(orbit_frames, "outputs/lego_orbit.gif", fps=20)


if __name__ == "__main__":
    main()