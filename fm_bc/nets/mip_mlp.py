"""MIP's residual MLP and its MLP observation encoder.

Ported from ``mip/networks/mlp.py`` (``MLP``, ``MLPResidualBlock``) and ``mip/encoders.py``
(``MLPEncoder``, ``Mlp``) of https://github.com/simchowitzlabpublic/much-ado-about-noising (MIT
License, Copyright (c) 2025 Chaoyi Pan). The defaults are the repo's state-observation config
(``network/mlp.yaml`` + ``network/_base.yaml``): width 512, 8 residual blocks, dropout 0.1, a
128-dim uniform-frequency time embedding and a 2-layer MLP encoder without label dropout. The
unused scalar output head is dropped.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["MIPEncoder", "MIPResidualMLP"]


class MIPEncoder(nn.Module):
    """``(B, To, obs_dim) -> (B, To, emb_dim)``: one MLP over the flattened observation window."""

    def __init__(self, obs_dim: int, To: int, emb_dim: int = 512, n_layers: int = 2):
        super().__init__()
        self.To, self.emb_dim = To, emb_dim
        dims = [obs_dim * To] + [emb_dim] * n_layers
        layers: list[nn.Module] = []
        for i, o in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(i, o), nn.LeakyReLU()]
        layers.append(nn.Linear(dims[-1], emb_dim * To))
        self.mlp = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp(obs.reshape(obs.shape[0], -1)).reshape(obs.shape[0], self.To, self.emb_dim)


class _ResidualBlock(nn.Module):
    """Pre-LayerNorm residual block, 4x expansion, GELU, orthogonal init."""

    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.activation = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.dropout2 = nn.Dropout(dropout)
        for lin in (self.linear1, self.linear2):
            nn.init.orthogonal_(lin.weight)
            nn.init.constant_(lin.bias, 0)

    def forward(self, x):
        h = self.dropout1(self.activation(self.linear1(self.norm1(x))))
        return x + self.dropout2(self.linear2(self.norm2(h)))


class MIPResidualMLP(nn.Module):
    """``(x (B, Ta, act_dim), s (B,), t (B,), condition (B, To, obs_dim)) -> (B, Ta, act_dim)``."""

    def __init__(self, act_dim: int, Ta: int, obs_dim: int, To: int, emb_dim: int = 512,
                 n_layers: int = 8, timestep_emb_dim: int = 128, max_freq: float = 100.0,
                 dropout: float = 0.1):
        super().__init__()
        if timestep_emb_dim % 2:
            raise ValueError("timestep_emb_dim must be even")
        self.act_dim = act_dim
        self.register_buffer("frequencies", torch.linspace(0, max_freq, timestep_emb_dim // 2))
        input_dim = act_dim * Ta + 2 * timestep_emb_dim + obs_dim * To
        self.input_proj = nn.Linear(input_dim, emb_dim)
        self.input_norm = nn.LayerNorm(emb_dim)
        self.input_activation = nn.GELU()
        self.residual_blocks = nn.ModuleList([_ResidualBlock(emb_dim, dropout)
                                              for _ in range(n_layers)])
        self.final_norm = nn.LayerNorm(emb_dim)
        self.main_output = nn.Linear(emb_dim, act_dim * Ta)
        for lin in (self.input_proj, self.main_output):
            nn.init.orthogonal_(lin.weight)
            nn.init.constant_(lin.bias, 0)

    def _embed_time(self, t: torch.Tensor) -> torch.Tensor:
        angles = t.unsqueeze(-1) * self.frequencies.unsqueeze(0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                condition: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x.flatten(1), self._embed_time(s), self._embed_time(t),
                       condition.flatten(1)], dim=-1)
        h = self.input_activation(self.input_norm(self.input_proj(h)))
        for block in self.residual_blocks:
            h = block(h)
        return self.main_output(self.final_norm(h)).view(x.shape[0], x.shape[1], self.act_dim)
