"""ACT's CVAE transformer -- the Action Chunking Transformer, state-only.

Ported from ``src/lerobot/policies/act/modeling_act.py`` of
https://github.com/huggingface/lerobot (Apache License 2.0), which is itself a port of
https://github.com/tonyzhaozh/act (MIT License, Copyright (c) 2023 Tony Z. Zhao) -- Zhao et al.
2023, "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware", arXiv:2304.13705 --
which is in turn DETR (facebookresearch/detr) with a CVAE latent bolted on. LeRobot's is the
implementation the community benchmarks against, so its defaults are the ones reproduced here:
``dim_model 512``, ``n_heads 8``, ``dim_feedforward 3200``, ``dropout 0.1``, ``latent_dim 32``,
4 encoder layers, 4 VAE-encoder layers and post-norm blocks with a ReLU feed-forward.

**``dec_layers`` defaults to 1, and that is faithful, not a simplification.** The released ACT
builds a 7-layer decoder with ``return_intermediate_dec=True`` and then reads
``self.transformer(...)[0]`` -- the *first* layer's output, not the last -- so layers 2-7 never
receive gradient. LeRobot's config carries the note verbatim: "Although the original ACT
implementation has 7 for ``n_decoder_layers``, there is a bug in the code that means only the
first layer is used. Here we match the original implementation by setting this to 1." One layer
is therefore both what produced the paper's numbers and what everyone benchmarks against.

Deviations from the source, all deliberate:

* **No ResNet-18 backbone and no per-camera feature tokens.** Both tasks in this repo are
  low-dimensional state, so the encoder's memory is ``1 + obs_steps`` tokens rather than the ~300
  a four-camera ACT flattens out of its feature maps. The architecture is unchanged; the
  information it was designed to route is not there. This is the same substitution the Diffusion
  Policy and MIP ports in this package already make.
* **``obs_steps`` observation tokens** where ACT has a single proprioception token (ACT is
  ``obs_steps = 1``). The learned 1-D position embedding widens from 2 entries to
  ``1 + obs_steps`` accordingly, and the CVAE encoder's sequence from ``2 + Tp`` to
  ``1 + To + Tp``.
* **The ``is_pad`` mask is dropped.** ACT masks edge-padded actions out of the CVAE encoder and
  computes an ``is_pad_head`` it never uses in the loss. :mod:`fm_bc.skate.chunks` edge-pads its
  windows and the other chunk baselines train on those padded targets, so ACT does too --
  symmetry with DP and MIP matters more here than matching ACT's masking.
* **Attention is ``nn.MultiheadAttention``** (DETR's own primitive, ``batch_first=True``), not the
  hand-rolled softmax of :mod:`fm_bc.nets.transformer`. That one exists because
  :func:`fm_bc.skate.policy.model_divergence` differentiates a backward pass
  (``create_graph=True``) and the fused kernels have no double backward -- but it runs only on the
  flow policies, never on a chunk baseline, so the constraint does not apply here. Reusing that
  module's blocks is also not possible: DETR adds the position embedding to the queries and keys
  but **not** the values, and those classes fuse q/k/v into one ``Linear`` over a single input.
* einops is replaced by ``expand`` / ``unsqueeze``, as in :mod:`fm_bc.nets.unet1d`.

Like the rest of :mod:`fm_bc.nets`, this imports torch and nothing else.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["ACT", "kl_divergence", "sinusoidal_pos_embedding"]


def sinusoidal_pos_embedding(num_positions: int, dim: int) -> torch.Tensor:
    """``(1, num_positions, dim)``, ACT's ``create_sinusoidal_pos_embedding``.

    Interleaved -- ``sin`` on the even channels, ``cos`` on the odd -- where
    :class:`fm_bc.nets.unet1d.SinusoidalPosEmb` concatenates the two halves. The two are a channel
    permutation of the same basis; this is ACT's.
    """
    pos = torch.arange(num_positions, dtype=torch.float64).unsqueeze(1)
    j = torch.arange(dim, dtype=torch.float64).unsqueeze(0)
    table = pos / torch.pow(10000.0, 2 * torch.div(j, 2, rounding_mode="floor") / dim)
    table[:, 0::2] = table[:, 0::2].sin()
    table[:, 1::2] = table[:, 1::2].cos()
    return table.float().unsqueeze(0)


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """``KL(q(z|.) || N(0, I))``, in ACT's reduction: summed over the latent dimension, meaned
    over the batch.

    The reduction is load-bearing. Meaning over the latent dimension too divides this by
    ``latent_dim`` (32), which silently turns ``kl_weight = 10`` into an effective 0.31, leaves
    the posterior unregularized and lets the CVAE route the action chunk through ``z`` -- which
    inference, at ``z = 0``, then cannot use.
    """
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp())).sum(-1).mean()


def _add_pos(x: torch.Tensor, pos: torch.Tensor | None) -> torch.Tensor:
    return x if pos is None else x + pos


class _EncoderLayer(nn.Module):
    """DETR's encoder layer: self-attention then a feed-forward, each residual and normalized."""

    def __init__(self, dim: int, nheads: int, dim_feedforward: int, dropout: float,
                 pre_norm: bool):
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attn = nn.MultiheadAttention(dim, nheads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.dropout1, self.dropout2 = nn.Dropout(dropout), nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = _add_pos(x, pos)  # the position goes to q and k, never to v
        x = skip + self.dropout1(self.self_attn(q, k, value=x)[0])
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = skip + self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        return x if self.pre_norm else self.norm2(x)


class _Encoder(nn.Module):
    """``n_layers`` encoder layers; the final norm exists only under pre-norm (DETR's
    ``encoder_norm = LayerNorm if normalize_before else None``)."""

    def __init__(self, n_layers: int, dim: int, nheads: int, dim_feedforward: int,
                 dropout: float, pre_norm: bool):
        super().__init__()
        self.layers = nn.ModuleList([
            _EncoderLayer(dim, nheads, dim_feedforward, dropout, pre_norm)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim) if pre_norm else nn.Identity()

    def forward(self, x: torch.Tensor, pos: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, pos)
        return self.norm(x)


class _DecoderLayer(nn.Module):
    """DETR's decoder layer: self-attention over the queries, cross-attention to the memory, then
    a feed-forward."""

    def __init__(self, dim: int, nheads: int, dim_feedforward: int, dropout: float,
                 pre_norm: bool):
        super().__init__()
        self.pre_norm = pre_norm
        self.self_attn = nn.MultiheadAttention(dim, nheads, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(dim, nheads, dropout=dropout,
                                                    batch_first=True)
        self.linear1 = nn.Linear(dim, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim)
        self.norm1, self.norm2, self.norm3 = (nn.LayerNorm(dim), nn.LayerNorm(dim),
                                              nn.LayerNorm(dim))
        self.dropout = nn.Dropout(dropout)
        self.dropout1, self.dropout2, self.dropout3 = (nn.Dropout(dropout), nn.Dropout(dropout),
                                                       nn.Dropout(dropout))
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                query_pos: torch.Tensor | None = None,
                memory_pos: torch.Tensor | None = None) -> torch.Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = _add_pos(x, query_pos)
        x = skip + self.dropout1(self.self_attn(q, k, value=x)[0])
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        # The memory's position embedding is re-added to the keys at every layer.
        x = skip + self.dropout2(self.multihead_attn(
            query=_add_pos(x, query_pos), key=_add_pos(memory, memory_pos), value=memory)[0])
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = skip + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(x)))))
        return x if self.pre_norm else self.norm3(x)


class _Decoder(nn.Module):
    """``n_layers`` decoder layers and a final LayerNorm, which DETR applies in both norm
    conventions."""

    def __init__(self, n_layers: int, dim: int, nheads: int, dim_feedforward: int,
                 dropout: float, pre_norm: bool):
        super().__init__()
        self.layers = nn.ModuleList([
            _DecoderLayer(dim, nheads, dim_feedforward, dropout, pre_norm)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, memory, query_pos=None, memory_pos=None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, query_pos=query_pos, memory_pos=memory_pos)
        return self.norm(x)


class ACT(nn.Module):
    """``(obs (B, To, obs_dim)[, actions (B, Tp, act_dim)]) -> chunk (B, Tp, act_dim), mu, logvar``.

    With ``actions`` the CVAE encoder runs and ``z`` is a reparameterized draw from
    ``q(z | obs, actions)``; without them ``z = 0`` -- the prior *mean*, which is what ACT uses at
    test time, **not** a draw from ``N(0, I)``. The model is therefore deterministic at inference,
    up to dropout: call ``eval()`` before sampling.
    """

    def __init__(self, obs_dim: int, act_dim: int, obs_steps: int = 2, horizon: int = 16,
                 hidden_dim: int = 512, dim_feedforward: int = 3200, nheads: int = 8,
                 enc_layers: int = 4, dec_layers: int = 1, vae_enc_layers: int = 4,
                 dropout: float = 0.1, latent_dim: int = 32, pre_norm: bool = False):
        super().__init__()
        self.obs_dim, self.act_dim = int(obs_dim), int(act_dim)
        self.obs_steps, self.horizon = int(obs_steps), int(horizon)
        self.hidden_dim, self.latent_dim = int(hidden_dim), int(latent_dim)

        # CVAE encoder: [CLS, To observation tokens, Tp action tokens] -> mu, logvar.
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.vae_obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.vae_action_proj = nn.Linear(act_dim, hidden_dim)
        self.vae_encoder = _Encoder(vae_enc_layers, hidden_dim, nheads, dim_feedforward, dropout,
                                    pre_norm)
        self.latent_proj = nn.Linear(hidden_dim, 2 * latent_dim)
        self.register_buffer(
            "vae_pos", sinusoidal_pos_embedding(1 + self.obs_steps + self.horizon, hidden_dim))

        # Encoder: [z token, To observation tokens] -> memory.
        self.latent_out_proj = nn.Linear(latent_dim, hidden_dim)
        self.obs_proj = nn.Linear(obs_dim, hidden_dim)
        self.memory_pos = nn.Embedding(1 + self.obs_steps, hidden_dim)
        self.encoder = _Encoder(enc_layers, hidden_dim, nheads, dim_feedforward, dropout,
                                pre_norm)

        # Decoder: Tp learned queries cross-attending to the memory -> the action chunk.
        self.query_embed = nn.Embedding(self.horizon, hidden_dim)
        self.decoder = _Decoder(dec_layers, hidden_dim, nheads, dim_feedforward, dropout,
                                pre_norm)
        self.action_head = nn.Linear(hidden_dim, act_dim)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """DETR's init, over the encoder and decoder stacks only -- not the projections, the
        embeddings or the VAE encoder, which keep torch's defaults (LeRobot's
        ``ACT._reset_parameters`` does exactly this)."""
        for p in [*self.encoder.parameters(), *self.decoder.parameters()]:
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _check(self, obs: torch.Tensor) -> None:
        if obs.ndim != 3 or obs.shape[1] != self.obs_steps or obs.shape[2] != self.obs_dim:
            raise ValueError(f"obs must be (B, {self.obs_steps}, {self.obs_dim}), got "
                             f"{tuple(obs.shape)}")

    def encode(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``(B, To, obs_dim), (B, Tp, act_dim) -> mu (B, latent_dim), logvar (B, latent_dim)``."""
        self._check(obs)
        if actions.shape[1] != self.horizon or actions.shape[2] != self.act_dim:
            raise ValueError(f"actions must be (B, {self.horizon}, {self.act_dim}), got "
                             f"{tuple(actions.shape)}")
        b = obs.shape[0]
        tokens = torch.cat([self.cls_embed.weight.unsqueeze(0).expand(b, -1, -1),
                            self.vae_obs_proj(obs), self.vae_action_proj(actions)], dim=1)
        cls_out = self.vae_encoder(tokens, self.vae_pos)[:, 0]  # the CLS token, not a pooling
        mu, logvar = self.latent_proj(cls_out).chunk(2, dim=-1)
        return mu, logvar

    def decode(self, obs: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """``(B, To, obs_dim), (B, latent_dim) -> chunk (B, Tp, act_dim)``."""
        self._check(obs)
        b = obs.shape[0]
        memory = self.encoder(
            torch.cat([self.latent_out_proj(z).unsqueeze(1), self.obs_proj(obs)], dim=1),
            self.memory_pos.weight.unsqueeze(0))
        # DETR's decoder input is zeros; the query embedding enters only as a position.
        queries = torch.zeros(b, self.horizon, self.hidden_dim, device=obs.device,
                              dtype=obs.dtype)
        out = self.decoder(queries, memory, query_pos=self.query_embed.weight.unsqueeze(0),
                           memory_pos=self.memory_pos.weight.unsqueeze(0))
        return self.action_head(out)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor | None = None,
                generator: torch.Generator | None = None):
        if actions is None:  # inference: the prior mean, not a sample
            z = torch.zeros(obs.shape[0], self.latent_dim, device=obs.device, dtype=obs.dtype)
            return self.decode(obs, z), None, None
        mu, logvar = self.encode(obs, actions)
        eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        z = mu + logvar.div(2).exp() * eps  # std = exp(logvar / 2), not exp(logvar)
        return self.decode(obs, z), mu, logvar
