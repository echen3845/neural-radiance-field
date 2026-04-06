import torch
import torch.nn as nn
import torch.nn.functional as F


class NeRF(nn.Module):
    def __init__(
        self,
        pos_in_dims=63,
        dir_in_dims=27,
        depth=8,
        width=256,
        skip_layer=4
    ):
        super().__init__()

        self.pos_in_dims = pos_in_dims
        self.dir_in_dims = dir_in_dims
        self.depth = depth
        self.width = width
        self.skip_layer = skip_layer

        self.pts_linears = nn.ModuleList()

        # First layer
        self.pts_linears.append(nn.Linear(pos_in_dims, width))

        # Hidden layers with one skip connection
        for i in range(1, depth):
            if i == skip_layer:
                self.pts_linears.append(nn.Linear(width + pos_in_dims, width))
            else:
                self.pts_linears.append(nn.Linear(width, width))

        # Output heads from position stream
        self.sigma_linear = nn.Linear(width, 1)
        self.feature_linear = nn.Linear(width, width)

        # Color branch
        self.view_linear = nn.Linear(width + dir_in_dims, width // 2)
        self.rgb_linear = nn.Linear(width // 2, 3)

    def forward(self, x, d):
        """
        Args:
            x: [..., pos_in_dims]
            d: [..., dir_in_dims]

        Returns:
            rgb: [..., 3]
            sigma: [..., 1]
        """
        h = x

        for i, layer in enumerate(self.pts_linears):
            if i == self.skip_layer:
                h = torch.cat([h, x], dim=-1)
            h = layer(h)
            h = F.relu(h)

        sigma = self.sigma_linear(h)
        features = self.feature_linear(h)

        h = torch.cat([features, d], dim=-1)
        h = self.view_linear(h)
        h = F.relu(h)

        rgb = self.rgb_linear(h)
        rgb = torch.sigmoid(rgb)  # constrain RGB to [0,1]

        sigma = F.softplus(sigma)

        return rgb, sigma