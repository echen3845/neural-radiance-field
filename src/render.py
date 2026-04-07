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

def sample_fine_points(rays_o, rays_d, z_vals_coarse, weights, n_fine):
    """
    Importance-sample fine points using coarse weight distribution (eq. 5 in paper).
    """
    N_rays, N_coarse = z_vals_coarse.shape

    # Build PDF/CDF from coarse weights
    weights = weights.detach() + 1e-5          # detach — no grad needed here
    pdf = weights / torch.sum(weights, dim=-1, keepdim=True)
    cdf = torch.cumsum(pdf, dim=-1)            # [N_rays, N_coarse]
    cdf = torch.cat(
        [torch.zeros_like(cdf[..., :1]), cdf], dim=-1
    )                                          # [N_rays, N_coarse+1]

    # Uniform samples
    u = torch.rand(N_rays, n_fine, device=z_vals_coarse.device).contiguous()

    # Invert CDF via searchsorted
    inds = torch.searchsorted(cdf.contiguous(), u, right=True)  # [N_rays, N_fine]

    # Clamp indices so gather never goes out of bounds
    # cdf has N_coarse+1 entries  → valid gather range [0, N_coarse]
    # bins (z_vals_coarse) has N_coarse entries → valid gather range [0, N_coarse-1]
    below = (inds - 1).clamp(min=0, max=N_coarse - 1)
    above = inds.clamp(min=0,       max=N_coarse - 1)

    # Gather CDF bounds — index into cdf (size N_coarse+1)
    # Use below+1 for the upper CDF bound so the interval is [cdf[below], cdf[below+1]]
    cdf_below = torch.gather(cdf, 1, below)           # [N_rays, N_fine]
    cdf_above = torch.gather(cdf, 1, above + 1)       # [N_rays, N_fine]  ← +1 safe: above <= N_coarse-1

    # Gather bin bounds from z_vals_coarse (size N_coarse)
    bins_below = torch.gather(z_vals_coarse, 1, below)
    bins_above = torch.gather(z_vals_coarse, 1, above)

    # Linear interpolation within each bin
    denom = (cdf_above - cdf_below).clamp(min=1e-5)
    t = (u - cdf_below) / denom
    z_vals_fine = bins_below + t * (bins_above - bins_below)

    return z_vals_fine  # [N_rays, N_fine]

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

def render_rays_hierarchical(
    coarse_model,
    fine_model,
    pos_encoder,
    dir_encoder,
    rays_o,
    rays_d,
    near,
    far,
    n_coarse,
    n_fine,
    perturb=False,
    white_background=True,
):
    """
    Full hierarchical render matching paper section 5.2.

    Returns:
        rgb_coarse, depth_coarse, acc_coarse  — for coarse loss
        rgb_fine,   depth_fine,   acc_fine    — final output
    """
    # ── coarse pass ─────────────────────────────────────────────
    pts_c, z_vals_c = sample_points_on_rays(
        rays_o, rays_d, near, far, n_coarse, perturb=perturb
    )
    dirs_c = rays_d[:, None, :].expand(-1, n_coarse, -1)
    enc_pts_c  = pos_encoder(pts_c)
    enc_dirs_c = dir_encoder(dirs_c)

    rgb_c, sigma_c = coarse_model(enc_pts_c, enc_dirs_c)
    rgb_map_c, depth_map_c, acc_map_c, weights_c = volume_render(
        rgb_c, sigma_c, z_vals_c, rays_d, white_background
    )

    # ── fine pass ────────────────────────────────────────────────
    z_vals_fine = sample_fine_points(rays_o, rays_d, z_vals_c, weights_c, n_fine)

    # Merge coarse + fine z_vals, sort along ray
    z_vals_all, _ = torch.sort(
        torch.cat([z_vals_c, z_vals_fine], dim=-1), dim=-1
    )  # [N_rays, N_coarse + N_fine]

    n_total = n_coarse + n_fine
    pts_f  = rays_o[:, None, :] + rays_d[:, None, :] * z_vals_all[:, :, None]
    dirs_f = rays_d[:, None, :].expand(-1, n_total, -1)

    enc_pts_f  = pos_encoder(pts_f)
    enc_dirs_f = dir_encoder(dirs_f)

    rgb_f, sigma_f = fine_model(enc_pts_f, enc_dirs_f)
    rgb_map_f, depth_map_f, acc_map_f, _ = volume_render(
        rgb_f, sigma_f, z_vals_all, rays_d, white_background
    )

    return (rgb_map_c, depth_map_c, acc_map_c), (rgb_map_f, depth_map_f, acc_map_f)

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