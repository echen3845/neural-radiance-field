import torch


def get_rays(H: int, W: int, focal: float, c2w: torch.Tensor):
    """
    Args:
        H, W: image height and width
        focal: focal length
        c2w: [4, 4] camera-to-world pose matrix

    Returns:
        rays_o: [H, W, 3]
        rays_d: [H, W, 3]
    """
    device = c2w.device

    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy"
    )

    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)  # [H, W, 3]

    # Rotate ray directions from camera frame to world frame
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)

    # Camera origin in world coordinates, broadcast to all pixels
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    return rays_o, rays_d

def get_rays_from_pixels(i, j, H, W, focal, c2w):
    """
    Args:
        i: [N] pixel x-coordinates
        j: [N] pixel y-coordinates
        H, W, focal: camera parameters
        c2w: [4, 4]

    Returns:
        rays_o: [N, 3]
        rays_d: [N, 3]
    """
    device = c2w.device
    i = i.to(device).float()
    j = j.to(device).float()

    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)  # [N, 3]

    rays_d = dirs @ c2w[:3, :3].T
    rays_o = c2w[:3, 3].expand_as(rays_d)

    return rays_o, rays_d