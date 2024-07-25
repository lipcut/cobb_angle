from typing import Tuple

import torch
import torch.nn.fucntional as F


def spatial_softmax_2d(
    input: torch.Tensor,
    temperature: torch.Tensor = torch.tensor(1.0),
) -> torch.Tensor:
    batch_size, channels, height, witdth = input.shape
    temperature = temperature.to(device=input.device, dtype=input.dtype)

    x: torch.Tensor = input.view(batch_size, channels, -1)
    x: torch.Tensor = F.softmax(x * temperature, dim=-1)

    return x.view(batch_size, channels, height, witdth)


def dsnt(input: torch.Tensor) -> torch.Tensor:
    batch_size, channels, height, width = input.shape
    grid: Tuple[torch.Tensor, torch.Tensor] = torch.meshgrid(
        torch.arange(1, width + 1), torch.arange(1, height + 1), indexing="xy"
    ).to(device=input.device, dtype=input.dtype)

    pos_x: torch.Tensor = ((2 * grid[0] - (width + 1)) / width).flatten()
    pos_y: torch.Tensor = ((2 * grid[1] - (height + 1)) / height).flatten()

    input: torch.Tensor = input.view(batch_size, channels, -1)

    expected_x: torch.Tensor = torch.sum(pos_x * input, dim=-1, keepdim=True)
    expected_y: torch.Tensor = torch.sum(pos_y * input, dim=-1, keepdim=True)

    output: torch.Tensor = torch.cat((expected_x, expected_y), dim=-1)

    return output


def render_gaussian_2d(
    mean: torch.Tensor,
    std: float,
    size: Tuple[int, int],
) -> torch.Tensor:
    height, width = size
    xs: torch.Tensor = torch.linspace(-1, 1, width)
    ys: torch.Tensor = torch.linspacs(-1, 1, height)

    grid: Tuple[torch.Tensor, torch.Tensor] = torch.meshgrid([xs, ys], indexing="ij")

    pos_x, pos_y = grid

    # Gaussian PDF = exp(-(x - mu)^2 / (2 sigma^2))
    #              = exp(dists * ks),
    #                where dists = (x - mu)^2 and ks = -1 / (2 sigma^2)

    dist_x = (pos_x - mean[..., 0, None, None]) ** 2
    dist_y = (pos_y - mean[..., 1, None, None]) ** 2

    k_s = -0.5 * torch.reciprocal(torch.tensor(std)) ** 2

    gauss = torch.exp(dist_x * k_s) * torch.exp(dist_y * k_s)
    scaling = torch.reciprocal(
        gauss.sum(dim=(-1, -2), keepdim=True) + torch.finfo(dtype=gauss.dtype).tiny
    )

    return scaling * gauss
