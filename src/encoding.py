import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, input_dims=3, num_freqs=10, include_input=True, log_sampling=True):
        super().__init__()
        self.input_dims = input_dims
        self.num_freqs = num_freqs
        self.include_input = include_input
        self.log_sampling = log_sampling

        if log_sampling:
            self.freq_bands = 2.0 ** torch.linspace(0.0, num_freqs - 1, steps=num_freqs)
        else:
            self.freq_bands = torch.linspace(1.0, 2.0 ** (num_freqs - 1), steps=num_freqs)

        self.out_dim = 0
        if include_input:
            self.out_dim += input_dims
        self.out_dim += input_dims * 2 * num_freqs

    def forward(self, x):
        """
        Args:
            x: [..., input_dims]

        Returns:
            encoded: [..., out_dim]
        """
        freq_bands = self.freq_bands.to(x.device)

        out = []
        if self.include_input:
            out.append(x)

        for freq in freq_bands:
            out.append(torch.sin(freq * x))
            out.append(torch.cos(freq * x))

        return torch.cat(out, dim=-1)