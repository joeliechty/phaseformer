"""Diffusion Policy's conditional 1D temporal UNet.

Ported from ``diffusion_policy/model/diffusion/{conditional_unet1d,conv1d_components,
positional_embedding}.py`` of https://github.com/real-stanford/diffusion_policy (MIT License,
Copyright (c) 2023 Columbia Artificial Intelligence and Robotics Lab). Changes: einops is
replaced by ``transpose`` / ``unsqueeze``, the unused local-condition branch is dropped, and an
optional second time input (``time2``) is embedded by its own sinusoid + MLP and concatenated to
the FiLM condition -- MeanFlow's ``(t, t - r)`` conditioning. With ``n_times=1`` the module is
the original ``ConditionalUnet1D`` with a global condition.
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

__all__ = ["ConditionalUnet1D", "SinusoidalPosEmb"]


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=x.device, dtype=x.dtype) * -emb)
        emb = x[:, None] * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Conv1dBlock(nn.Module):
    """Conv1d -> GroupNorm -> Mish."""

    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Two conv blocks with FiLM modulation (per-channel scale and bias) from ``cond``."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int, kernel_size: int = 3,
                 n_groups: int = 8, cond_predict_scale: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
        ])
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))
        self.residual_conv = (nn.Conv1d(in_channels, out_channels, 1)
                              if in_channels != out_channels else nn.Identity())

    def forward(self, x, cond):
        """``x`` ``(B, C_in, T)``, ``cond`` ``(B, cond_dim)`` -> ``(B, C_out, T)``."""
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.cond_predict_scale:
            embed = embed.reshape(embed.shape[0], 2, self.out_channels, 1)
            out = embed[:, 0] * out + embed[:, 1]
        else:
            out = out + embed
        out = self.blocks[1](out)
        return out + self.residual_conv(x)


def _time_encoder(dsed: int) -> nn.Sequential:
    return nn.Sequential(SinusoidalPosEmb(dsed), nn.Linear(dsed, dsed * 4), nn.Mish(),
                         nn.Linear(dsed * 4, dsed))


class ConditionalUnet1D(nn.Module):
    """``(sample (B, T, input_dim), timestep (B,), global_cond (B, G)) -> (B, T, input_dim)``.

    ``n_times=2`` adds a second time input (``time2``) with its own encoder.
    """

    def __init__(self, input_dim: int, global_cond_dim: int | None = None,
                 diffusion_step_embed_dim: int = 256, down_dims: Sequence[int] = (256, 512, 1024),
                 kernel_size: int = 3, n_groups: int = 8, cond_predict_scale: bool = False,
                 n_times: int = 1):
        super().__init__()
        if n_times not in (1, 2):
            raise ValueError(f"n_times must be 1 or 2, got {n_times}")
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = _time_encoder(dsed)
        self.time2_encoder = _time_encoder(dsed) if n_times == 2 else None
        cond_dim = dsed * n_times + (global_cond_dim or 0)
        in_out = list(zip(all_dims[:-1], all_dims[1:]))

        def block(i, o):
            return ConditionalResidualBlock1D(i, o, cond_dim=cond_dim, kernel_size=kernel_size,
                                              n_groups=n_groups,
                                              cond_predict_scale=cond_predict_scale)

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([block(mid_dim, mid_dim), block(mid_dim, mid_dim)])

        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(nn.ModuleList([
                block(dim_in, dim_out), block(dim_out, dim_out),
                Downsample1d(dim_out) if not is_last else nn.Identity(),
            ]))

        self.up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(nn.ModuleList([
                block(dim_out * 2, dim_in), block(dim_in, dim_in),
                Upsample1d(dim_in) if not is_last else nn.Identity(),
            ]))

        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor,
                global_cond: torch.Tensor | None = None,
                time2: torch.Tensor | None = None) -> torch.Tensor:
        x = sample.transpose(1, 2)  # (B, input_dim, T)
        timestep = torch.as_tensor(timestep, device=x.device)
        feats = [self.diffusion_step_encoder(timestep.to(x.dtype).expand(x.shape[0]))]
        if self.time2_encoder is not None:
            if time2 is None:
                raise ValueError("this UNet takes a second time input (time2)")
            feats.append(self.time2_encoder(torch.as_tensor(time2, device=x.device)
                                            .to(x.dtype).expand(x.shape[0])))
        if global_cond is not None:
            feats.append(global_cond)
        global_feature = torch.cat(feats, dim=-1)

        h = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, h.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        return self.final_conv(x).transpose(1, 2)
