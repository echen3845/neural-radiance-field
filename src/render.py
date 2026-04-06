import torch

def sample_points_on_rays(rays_o, rays_d, near, far, n_samples, perturb=False):
    """
    Args:
        rays_0: [H, W, 3] ray origins
        rays_d: [H, W, 3] ray directions
        near: float, near plane distance
        far: float, far plane distance
        n_samples: int, number of samples per ray
        perturb: bool, whether to add random perturbation to sample positions

    Returns:
        points: [H, W, n_samples, 3] sampled points along rays
    """
    device = rays_o.device

    t_vals = torch.linspace(0.0, 1.0, steps=n_samples, device=device)
    z_vals = near * (1.0 - t_vals) + far * t_vals  # [n_samples]

    # Broadcast z_vals to match the ray shape
    while len(z_vals.shape) < len(rays_o.shape):
        z_vals = z_vals.unsqueeze(0)

    z_vals = z_vals.expand(*rays_o.shape[:-1], n_samples)

    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand(z_vals.shape, device=device)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    return pts, z_vals

def volume_render(rgb, sigma, z_vals, rays_d, white_background=True):
    """
    Perform differentiable volume rendering.

    Args:
        rgb: [N_rays, N_samples, 3]
        sigma: [N_rays, N_samples, 1]
        z_vals: [N_rays, N_samples]
        rays_d: [N_rays, 3]
        white_background: bool

    Returns:
        rgb_map: [N_rays, 3]
        depth_map: [N_rays]
        acc_map: [N_rays]
        weights: [N_rays, N_samples]
    """
    sigma = sigma.squeeze(-1)  # [N_rays, N_samples]

    # Distance between consecutive samples
    dists = z_vals[..., 1:] - z_vals[..., :-1]  # [N_rays, N_samples-1]

    # Last interval: make it very large so the ray "ends"
    infinity_pad = torch.full_like(dists[..., :1], 1e10)
    dists = torch.cat([dists, infinity_pad], dim=-1)  # [N_rays, N_samples]

    # Account for ray direction magnitude
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Alpha values
    alpha = 1.0 - torch.exp(-sigma * dists)  # [N_rays, N_samples]

    # Transmittance
    eps = 1e-10
    transmittance = torch.cumprod(
        torch.cat([
            torch.ones((alpha.shape[0], 1), device=alpha.device),
            1.0 - alpha + eps
        ], dim=-1),
        dim=-1
    )[:, :-1]

    # Weights
    weights = transmittance * alpha  # [N_rays, N_samples]

    # Final rendered outputs
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)   # [N_rays, 3]
    depth_map = torch.sum(weights * z_vals, dim=-1)         # [N_rays]
    acc_map = torch.sum(weights, dim=-1)                    # [N_rays]

    if white_background:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, depth_map, acc_map, weights

def render_rays(
    model,
    pos_encoder,
    dir_encoder,
    rays_o,
    rays_d,
    near,
    far,
    n_samples,
    perturb=False,
    white_background=True
):
    """
    Render a batch of rays.

    Args:
        model: NeRF model
        pos_encoder: positional encoder for 3D points
        dir_encoder: positional encoder for directions
        rays_o: [N_rays, 3]
        rays_d: [N_rays, 3]

    Returns:
        rgb_map: [N_rays, 3]
        depth_map: [N_rays]
        acc_map: [N_rays]
    """
    pts, z_vals = sample_points_on_rays(
        rays_o, rays_d, near, far, n_samples, perturb=perturb
    )  # [N_rays, N_samples, 3], [N_rays, N_samples]

    dirs = rays_d[:, None, :].expand(-1, n_samples, -1)

    encoded_pts = pos_encoder(pts)
    encoded_dirs = dir_encoder(dirs)

    rgb, sigma = model(encoded_pts, encoded_dirs)

    rgb_map, depth_map, acc_map, _ = volume_render(
        rgb, sigma, z_vals, rays_d, white_background=white_background
    )

    return rgb_map, depth_map, acc_map