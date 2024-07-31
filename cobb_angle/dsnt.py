from typing import Tuple

import torch
import torch.nn.functional as F


def spatial_softmax_2d(
    input: torch.Tensor,
    temperature: torch.Tensor = torch.tensor(1.0),
) -> torch.Tensor:
    batch_size, channels, height, width = input.shape
    temperature = temperature.to(device=input.device, dtype=input.dtype)

    x: torch.Tensor = input.view(batch_size, channels, -1)
    x: torch.Tensor = F.softmax(x * temperature, dim=-1)

    return x.view(batch_size, channels, height, width)


def dsnt(input: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = input.shape
    grid: Tuple[torch.Tensor, torch.Tensor] = torch.meshgrid(
        torch.arange(1, width + 1), torch.arange(1, height + 1), indexing="xy"
    )

    pos_x: torch.Tensor = ((2 * grid[0] - (width + 1)) / width).flatten()
    pos_y: torch.Tensor = ((2 * grid[1] - (height + 1)) / height).flatten()
    pos_x: torch.Tensor = pos_x.to(device=input.device, dtype=input.dtype)
    pos_y: torch.Tensor = pos_y.to(device=input.device, dtype=input.dtype)

    input: torch.Tensor = input.view(batch_size, channels, -1)

    expected_x: torch.Tensor = torch.sum(pos_x * input, dim=-1, keepdim=True)
    expected_y: torch.Tensor = torch.sum(pos_y * input, dim=-1, keepdim=True)

    output: torch.Tensor = torch.cat((expected_x, expected_y), dim=-1)

    return output


def render_gaussian_2d(
    mean: torch.Tensor,
    size: Tuple[int, int],
    std: float = 1.0,
) -> torch.Tensor:
    height, width = size
    xs: torch.Tensor = torch.linspace(-1, 1, width).to(mean.device)
    ys: torch.Tensor = torch.linspace(-1, 1, height).to(mean.device)

    grid: Tuple[torch.Tensor, torch.Tensor] = torch.meshgrid([xs, ys], indexing="xy")

    pos_x, pos_y = grid

    # Gaussian PDF = exp(-(x - mu)^2 / (2 sigma^2))
    #              = exp(dists * ks),
    #                where dists = (x - mu)^2 and ks = -1 / (2 sigma^2)

    dist_x: torch.Tensor = (pos_x - mean[..., 0, None, None]) ** 2
    dist_y: torch.Tensor = (pos_y - mean[..., 1, None, None]) ** 2

    kx_s: torch.Tensor = -0.5 * torch.reciprocal(torch.tensor(std) / (width / 2)) ** 2
    ky_s: torch.Tensor = -0.5 * torch.reciprocal(torch.tensor(std) / (height / 2)) ** 2

    gauss: torch.Tensor = torch.exp(dist_x * kx_s) * torch.exp(dist_y * ky_s)
    scaling: torch.Tensor = torch.reciprocal(
        torch.clamp(gauss.sum(dim=(-1, -2), keepdim=True), min=1e-32)
    )
    breakpoint()

    return scaling * gauss
