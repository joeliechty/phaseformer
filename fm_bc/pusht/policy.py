"""State-feedback policies for Push-T, with and without a phase input, and the agent score.

The Push-T counterpart of :class:`fm_bc.skate.policy.FlowMLP`: the same three arms the skate study
compares -- the state-only MLP, MLP+phase and MLP+phase+score -- on the other task.

Two backbones implement those arms, selected by :data:`ARCHS` / :func:`make_pusht_policy` and by
``PolicyConfig.arch``. :class:`PushTMLP` concatenates ``2 rho - 1`` to its input vector;
:class:`PushTTransformer` injects the same ``rho`` through adaptive layer norm at every
normalization instead (:mod:`fm_bc.nets.transformer`). They have the same call signature, the same
checkpoint contract and the same target, so an arm is a backbone choice and nothing else.

**The output is the phase velocity of the agent position**, ``u = dx/drho``: the derivative of
the pusher's xy with respect to phase, in box-normalized units per unit phase. On a
demonstration it is the forward difference over one step's phase increment,

    u_k = (L - 1) (x_{k+1} - x_k),     so     x_{k+1} = x_k + u_k * drho,   drho = 1 / (L - 1),

the discrete form of the flow ``dx/drho = u(x, rho)`` the paper's math is written for, and the
Push-T analogue of the skate rollout ``x_{k+1} = x_k (+) dt * u``. It is the *achieved* motion
(``PushTEpisodes.agent_delta``, edge-held at each episode's last step), not the commanded target
in the dataset's ``action`` column.

**Inputs are ``[o, x, rho]``**: the object observation ``o`` -- the T-block pose as box-normalized
xy and ``cos``/``sin`` of its heading -- the agent position ``x`` (box-normalized xy) and
``2 rho - 1``, all at the current step. With ``obs_steps > 1`` the earlier steps' ``[o, x]`` are
prepended, but the default is the current step only, which is the conditioning the math is
written for, ``u(x, rho | o)``: past positions would let the network read off the previous
displacement instead of learning the field.

**The score is of the agent's position density, not of the full state**, because ``u`` lives in
the agent's tangent space: ``b = (u - u_hat) . grad log p_rho(x)`` pairs the phase-velocity error
with the score of the xy marginal at the same phase (Euclidean, D = 2). The block pose is
observed but not a flow coordinate. Both factors are in box-normalized units, so ``b`` is per
unit phase.

Everything here imports torch and the Push-T subtree only (see
``tests/pusht/test_pusht_data.py::test_pusht_imports_without_theseus``). :func:`agent_score`
restates ``fm_bc.divergence.score_kernel_at(metric="xy")`` for that reason --
``fm_bc.divergence`` pulls in scipy at import -- and a test pins the two to each other.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from ..nets.transformer import PhaseTransformer
from .chunks import ACT_DIM, OBS_DIM, embed_state, tee_points
from .constants import N_TEE_POINTS, TEE_POINTS
from .normalize import Normalizer

__all__ = ["ARCHS", "PushTBigTransformer", "PushTLargeMLP", "PushTLargeTransformer", "PushTMLP",
           "PushTMediumDeepMLP", "PushTMediumDeepTransformer", "PushTMediumMLP",
           "PushTMediumTransformer",
           "PushTTransformer",
           "agent_score", "make_pusht_policy", "load_pusht_policy"]

# Mirrors fm_bc.skate.policy._ACTIVATIONS, which cannot be imported without theseus.
_ACTIVATIONS = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


class PushTMLP(nn.Module):
    """``(state window (B, To, 5), phase (B,)) -> agent phase velocity dx/drho (B, 2)``.

    Input is ``[o, x]`` of each observed step, flattened -- block ``[xy_n, cos, sin]`` then agent
    ``xy_n`` -- plus ``2 rho - 1`` when ``use_phase``, the phase convention of ``FlowMLP``. With
    ``use_phase=False`` the phase is still accepted and ignored, so every arm has one call
    signature.
    """

    kind: str = "pusht_mlp"

    def __init__(
        self,
        pos_norm: Normalizer | dict,
        obs_steps: int = 1,
        hidden: Sequence[int] = (512, 512, 512, 512),
        activation: str = "silu",
        use_phase: bool = True,
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {tuple(_ACTIVATIONS)}, got {activation!r}")
        if obs_steps < 1:
            raise ValueError(f"obs_steps must be >= 1, got {obs_steps}")
        if isinstance(pos_norm, dict):
            pos_norm = Normalizer.from_dict(pos_norm)
        scale = torch.as_tensor(pos_norm.scale, dtype=torch.float32).flatten()
        offset = torch.as_tensor(pos_norm.offset, dtype=torch.float32).flatten()
        if scale.numel() != 2:
            raise ValueError(f"pos_norm must be the 2-D agent_pos normalizer, got {scale.numel()}-D")
        self.obs_steps, self.use_phase = int(obs_steps), bool(use_phase)
        self.hidden, self.activation = tuple(int(h) for h in hidden), activation
        self.register_buffer("pos_scale", scale)
        self.register_buffer("pos_offset", offset)

        layers: list[nn.Module] = []
        width = self.obs_steps * OBS_DIM + int(self.use_phase)
        for h in self.hidden:
            layers += [nn.Linear(width, h), _ACTIVATIONS[activation]()]
            width = h
        layers.append(nn.Linear(width, ACT_DIM))
        self.net = nn.Sequential(*layers)

    @property
    def pos(self) -> Normalizer:
        return Normalizer(scale=self.pos_scale, offset=self.pos_offset)

    def embed(self, obs: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        e = embed_state(obs, self.pos)  # (B, To, 6) [agent_xy, block_xy, cos, sin]
        x = torch.cat([e[..., 2:6], e[..., 0:2]], dim=-1).flatten(1)  # [o, x] per step
        if self.use_phase:
            x = torch.cat([x, 2.0 * phase[:, None].to(x) - 1.0], dim=-1)
        return x

    def forward(self, obs: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Phase velocity ``dx/drho`` of the agent, box-normalized units per unit phase, ``(B, 2)``."""
        return self.net(self.embed(obs, phase))

    @torch.no_grad()
    def next_position(self, obs: torch.Tensor, phase: torch.Tensor,
                      dphase: torch.Tensor) -> torch.Tensor:
        """One Euler step of the flow, ``x + u * drho``, in pixels, ``(B, 2)``.

        ``dphase`` is the phase increment per control step, ``1 / (L - 1)`` on a demonstration;
        in a rollout it has to come from a phase-rate estimate.
        """
        x = self.pos.apply(obs[:, -1, :2])
        return self.pos.invert(x + self(obs, phase) * dphase[:, None].to(x))

    def config(self) -> dict:
        return {
            "pos_norm": {"scale": self.pos_scale.tolist(), "offset": self.pos_offset.tolist()},
            "obs_steps": self.obs_steps, "hidden": list(self.hidden),
            "activation": self.activation, "use_phase": self.use_phase,
        }

    def save(self, path: str | Path, **extra) -> None:
        torch.save({"kind": self.kind, "config": self.config(),
                    "state_dict": self.state_dict(), **extra}, path)


class PushTMediumMLP(PushTMLP):
    """:class:`PushTMLP` widened to the medium transformer's budget: four hidden layers of 768
    instead of 512, **1,779,458 parameters** (1,778,690 without the phase input) against
    :class:`PushTMediumTransformer`'s 1,791,002 -- so the two backbones stay parameter-matched at
    this size as the published pair is at theirs. A separate class with its own run suffix
    (``_mlpmed``), so the published MLP's defaults and run names are untouched.
    """

    kind: str = "pusht_mlp_medium"

    def __init__(self, pos_norm: Normalizer | dict, obs_steps: int = 1,
                 hidden: Sequence[int] = (768,) * 4, activation: str = "silu",
                 use_phase: bool = True):
        super().__init__(pos_norm, obs_steps, hidden, activation, use_phase)


class PushTMediumDeepMLP(PushTMLP):
    """:class:`PushTMLP` with the medium budget spent on depth: eight hidden layers of 512, twice
    the published four, as :class:`PushTMediumDeepTransformer` doubles the transformer's blocks.
    **1,843,714 parameters** (1,843,202 without the phase input): 1.04x :class:`PushTMediumMLP`
    and 1.07x the medium-deep transformer's 1,730,818. Run suffix ``_mlpdeep``.
    """

    kind: str = "pusht_mlp_medium_deep"

    def __init__(self, pos_norm: Normalizer | dict, obs_steps: int = 1,
                 hidden: Sequence[int] = (512,) * 8, activation: str = "silu",
                 use_phase: bool = True):
        super().__init__(pos_norm, obs_steps, hidden, activation, use_phase)


class PushTLargeMLP(PushTMLP):
    """:class:`PushTMediumMLP` made deeper: seven hidden layers of 768, **3,551,234 parameters**
    (3,550,466 without the phase input) against :class:`PushTLargeTransformer`'s 3,416,762 (1.04x).
    As the large transformer is the medium one plus depth, this is the medium MLP plus depth;
    seven layers rather than eight keeps it matched to the transformer. Run suffix ``_mlplarge``.
    """

    kind: str = "pusht_mlp_large"

    def __init__(self, pos_norm: Normalizer | dict, obs_steps: int = 1,
                 hidden: Sequence[int] = (768,) * 7, activation: str = "silu",
                 use_phase: bool = True):
        super().__init__(pos_norm, obs_steps, hidden, activation, use_phase)


class PushTTransformer(nn.Module):
    """``(state window (B, To, 5), phase (B,)) -> agent phase velocity dx/drho (B, 2)``.

    :class:`PushTMLP`'s interface and target with an AdaLN transformer in place of the MLP: the
    observation becomes context tokens, learned query tokens read them by cross-attention, and the
    phase drives the modulation at every normalization
    (:class:`fm_bc.nets.transformer.PhaseTransformer`). ``use_phase=False`` still accepts and
    ignores the phase, so every arm has one call signature.

    **The observation is tokenized per entity per step**: a block token ``[xy_n, cos, sin]`` and an
    agent token ``[xy_n]`` for each of the ``To`` observed steps, each with its own projection and a
    learned step embedding, so ``M = 2 To``. One token per step would make ``M = 1`` at the default
    ``obs_steps=1``, where a softmax over a single key is the identity and cross-attention does
    nothing.

    Deterministic, exactly as :class:`PushTMLP` is: no noise input and no sampling, so
    ``u = f(obs, rho)``. ``dropout`` defaults to 0 for that reason -- it would otherwise make the
    *training* loss stochastic where the MLP arms' is not (inference is unaffected either way, since
    every rollout and evaluation path calls ``eval()`` first).

    ``horizon > 1`` predicts ``u_k .. u_{k+horizon-1}``, ``(B, horizon, 2)``; the default
    ``horizon=1`` returns ``(B, 2)`` and is the drop-in.

    **The defaults are roughly parameter-matched to** :class:`PushTMLP`: ``hidden_dim=128,
    num_layers=2`` is 0.91M parameters against the MLP default ``(512,) * 4``'s 0.79M (1.14x), so
    an arm comparison isolates the backbone rather than the budget. Scaling up gets expensive
    fast -- ``hidden_dim=128, num_layers=4`` is 1.7M and ``hidden_dim=256, num_layers=4`` is 5.5M.
    Each run records its own count as ``n_params`` in ``config.json``.
    """

    kind: str = "pusht_transformer"
    _BLOCK_DIM, _AGENT_DIM = 4, 2  # [xy_n, cos, sin] and [xy_n] of the 6-D embedded state

    def __init__(
        self,
        pos_norm: Normalizer | dict,
        obs_steps: int = 1,
        horizon: int = 1,
        use_phase: bool = True,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        phase_dim: int = 128,
    ):
        super().__init__()
        if obs_steps < 1:
            raise ValueError(f"obs_steps must be >= 1, got {obs_steps}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        if isinstance(pos_norm, dict):
            pos_norm = Normalizer.from_dict(pos_norm)
        scale = torch.as_tensor(pos_norm.scale, dtype=torch.float32).flatten()
        offset = torch.as_tensor(pos_norm.offset, dtype=torch.float32).flatten()
        if scale.numel() != 2:
            raise ValueError(f"pos_norm must be the 2-D agent_pos normalizer, got {scale.numel()}-D")
        self.obs_steps, self.horizon = int(obs_steps), int(horizon)
        self.use_phase = bool(use_phase)
        self.hidden_dim, self.num_layers = int(hidden_dim), int(num_layers)
        self.num_heads, self.mlp_ratio = int(num_heads), float(mlp_ratio)
        self.dropout, self.phase_dim = float(dropout), int(phase_dim)
        self.register_buffer("pos_scale", scale)
        self.register_buffer("pos_offset", offset)

        self.block_proj = nn.Linear(self._BLOCK_DIM, self.hidden_dim)
        self.agent_proj = nn.Linear(self._AGENT_DIM, self.hidden_dim)
        # Which observed step a token came from, and whether it is the block or the agent.
        self.step_emb = nn.Parameter(torch.randn(1, self.obs_steps, 1, self.hidden_dim) * 0.02)
        self.net = self._make_net()
        for lin in (self.block_proj, self.agent_proj):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def _make_net(self) -> PhaseTransformer:
        """The backbone this policy wraps -- overridden by subclasses that want it built
        differently. Called from ``__init__`` once every field it reads is set."""
        return PhaseTransformer(
            out_dim=ACT_DIM, n_tokens=self.horizon, hidden_dim=self.hidden_dim,
            num_layers=self.num_layers, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
            dropout=self.dropout, phase_dim=self.phase_dim, use_phase=self.use_phase,
        )

    @property
    def pos(self) -> Normalizer:
        return Normalizer(scale=self.pos_scale, offset=self.pos_offset)

    @property
    def n_context_tokens(self) -> int:
        """``M``, the number of tokens cross-attention reads -- see :meth:`tokens`."""
        return 2 * self.obs_steps

    def tokens(self, obs: torch.Tensor) -> torch.Tensor:
        """``(B, To, 5)`` raw states -> ``(B, 2 To, hidden_dim)`` context tokens."""
        e = embed_state(obs, self.pos)  # (B, To, 6) [agent_xy, block_xy, cos, sin]
        block = self.block_proj(e[..., 2:6])  # object pose, as in PushTMLP's [o, x] order
        agent = self.agent_proj(e[..., 0:2])
        pair = torch.stack([block, agent], dim=2) + self.step_emb  # (B, To, 2, hidden_dim)
        return pair.flatten(1, 2)

    def forward(self, obs: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Phase velocity ``dx/drho`` of the agent, box-normalized units per unit phase, ``(B, 2)``
        (a chunk ``(B, horizon, 2)`` if chunked)."""
        u = self.net(self.tokens(obs), phase.reshape(-1).to(self.pos_scale))
        return u[:, 0] if self.horizon == 1 else u

    @torch.no_grad()
    def next_position(self, obs: torch.Tensor, phase: torch.Tensor,
                      dphase: torch.Tensor) -> torch.Tensor:
        """One Euler step of the flow, ``x + u * drho``, in pixels, ``(B, 2)`` -- see
        :meth:`PushTMLP.next_position`."""
        x = self.pos.apply(obs[:, -1, :2])
        u = self(obs, phase)
        if self.horizon > 1:
            u = u[:, 0]
        return self.pos.invert(x + u * dphase[:, None].to(x))

    def config(self) -> dict:
        return {
            "pos_norm": {"scale": self.pos_scale.tolist(), "offset": self.pos_offset.tolist()},
            "obs_steps": self.obs_steps, "horizon": self.horizon, "use_phase": self.use_phase,
            "hidden_dim": self.hidden_dim, "num_layers": self.num_layers,
            "num_heads": self.num_heads, "mlp_ratio": self.mlp_ratio, "dropout": self.dropout,
            "phase_dim": self.phase_dim,
        }

    def save(self, path: str | Path, **extra) -> None:
        torch.save({"kind": self.kind, "config": self.config(),
                    "state_dict": self.state_dict(), **extra}, path)


class PushTMediumTransformer(PushTTransformer):
    """:class:`PushTTransformer` at twice the parameters, and nothing else changed.

    The same tokenization (block and agent tokens), the same self- and cross-attention, the same
    two blocks and four heads -- only wider. ``hidden_dim`` goes 128 -> 180 (~sqrt(2) x, rounded to
    a multiple of the four heads) and ``phase_dim`` tracks it as it does at the published size, for
    **1,791,002 parameters** against 907,522 (1.97x). A scale-only test of the published backbone,
    where :class:`PushTBigTransformer` also changes the design.

    A separate class for the same reason as the big one: the published backbone's defaults and run
    names stay untouched, and ``--arch transformer_medium`` needs no other flags. Run names take
    ``_tfmed``.
    """

    kind: str = "pusht_transformer_medium"

    def __init__(
        self,
        pos_norm: Normalizer | dict,
        obs_steps: int = 1,
        horizon: int = 1,
        use_phase: bool = True,
        hidden_dim: int = 180,
        num_layers: int = 2,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        phase_dim: int = 180,
    ):
        super().__init__(
            pos_norm=pos_norm, obs_steps=obs_steps, horizon=horizon, use_phase=use_phase,
            hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, phase_dim=phase_dim,
        )


class PushTMediumDeepTransformer(PushTTransformer):
    """:class:`PushTTransformer` at twice the parameters by depth: four blocks instead of two.

    :class:`PushTMediumTransformer`'s depth-for-width twin -- ``hidden_dim``, ``phase_dim`` and the
    four heads stay at the published 128, and ``num_layers`` goes 2 -> 4, for **1,730,818
    parameters** (1.91x). Run names take ``_tfdeep``.
    """

    kind: str = "pusht_transformer_medium_deep"

    def __init__(
        self,
        pos_norm: Normalizer | dict,
        obs_steps: int = 1,
        horizon: int = 1,
        use_phase: bool = True,
        hidden_dim: int = 128,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        phase_dim: int = 128,
    ):
        super().__init__(
            pos_norm=pos_norm, obs_steps=obs_steps, horizon=horizon, use_phase=use_phase,
            hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, phase_dim=phase_dim,
        )


class PushTLargeTransformer(PushTTransformer):
    """:class:`PushTTransformer` with both medium changes at once: wider *and* deeper.

    ``hidden_dim = phase_dim = 180`` (:class:`PushTMediumTransformer`'s width) and four blocks
    (:class:`PushTMediumDeepTransformer`'s depth), heads and tokenization unchanged, for
    **3,416,762 parameters** (3.76x the published 907,522). Run names take ``_tflarge``.
    """

    kind: str = "pusht_transformer_large"

    def __init__(
        self,
        pos_norm: Normalizer | dict,
        obs_steps: int = 1,
        horizon: int = 1,
        use_phase: bool = True,
        hidden_dim: int = 180,
        num_layers: int = 4,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        phase_dim: int = 180,
    ):
        super().__init__(
            pos_norm=pos_norm, obs_steps=obs_steps, horizon=horizon, use_phase=use_phase,
            hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, phase_dim=phase_dim,
        )


class PushTBigTransformer(PushTTransformer):
    """:class:`PushTTransformer` at ~67M parameters, with cross-attention as its only attention.

    Same interface, same target, same checkpoint contract -- a separate class rather than options
    on the 0.87M model so that the published backbone's defaults, module tree and run names are
    untouched. Three things differ, and each one follows from cross-attention being the mechanism
    this backbone is scaled around.

    **The block is tokenized as points on it, not as a pose.** ``PushTTransformer`` emits two
    context tokens per observed step -- the block's ``[xy, cos, sin]`` and the agent's ``[xy]`` --
    so at the default ``obs_steps=1`` cross-attention chooses between two keys, and widening the
    network buys value and output projections rather than a richer attention pattern. Here the
    T's eight vertices and its two rectangle centroids ride along as their own tokens
    (:func:`fm_bc.pusht.chunks.tee_points`), for ``M = 12`` at ``obs_steps=1``. Those points are a
    fixed rigid function of the same ``(block_x, block_y, theta)``, so **the observation carries
    exactly the information it did before**; what changes is that the query can attend to the stem
    tip or a crossbar corner separately instead of to one summary of the pose. The pose and agent
    tokens are kept in slots 0 and 1, so this tokenization is a strict superset of the parent's.

    **There is no self-attention.** With ``horizon=1`` there is a single query token, and a softmax
    over one key is the constant 1 however the query is projected -- so self-attention's q and k
    receive exactly zero gradient, and its remaining path is a linear gated residual the block's
    MLP already subsumes (see :class:`fm_bc.nets.transformer.PhaseTransformer`). Dropping it
    frees ``4 d^2 + 3 d phase_dim`` per block. At this shape ``num_layers=8`` without it costs the
    same parameters and the same FLOPs as ``num_layers=6`` with it (``8 * 12 d^2 = 6 * 16 d^2``),
    so the budget buys two more live blocks instead of ~7M dead weights.

    **``phase_dim`` does not track the width.** Each adaLN site is ``Linear(phase_dim, 3 d)``, so
    letting ``phase_dim = hidden_dim`` would spend 37% of the model transforming a scalar; held at
    256 the modulation is ~17% and cross-attention ~21%.

    Defaults are the configuration meant to be trained -- ``hidden_dim=768, num_layers=8,
    num_heads=12, phase_dim=256``, **66,877,698 parameters** -- so ``--arch transformer_big`` needs
    no other flags. Run names take ``_tfbig``, so nothing shares a directory with ``_tf``.
    """

    kind: str = "pusht_transformer_big"
    _POINT_DIM = 2  # a T point token is its normalized xy

    def __init__(
        self,
        pos_norm: Normalizer | dict,
        obs_steps: int = 1,
        horizon: int = 1,
        use_phase: bool = True,
        hidden_dim: int = 768,
        num_layers: int = 8,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        phase_dim: int = 256,
    ):
        super().__init__(
            pos_norm=pos_norm, obs_steps=obs_steps, horizon=horizon, use_phase=use_phase,
            hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads,
            mlp_ratio=mlp_ratio, dropout=dropout, phase_dim=phase_dim,
        )
        # Not persistent: it is a constant of the environment, so a checkpoint should take it from
        # the code it is loaded by rather than carry a copy that can silently go stale.
        self.register_buffer("tee_local", torch.tensor(TEE_POINTS, dtype=torch.float32),
                             persistent=False)
        self.point_proj = nn.Linear(self._POINT_DIM, self.hidden_dim)
        # Which point of the T a token is. The projection is shared, so without this the point
        # tokens are permutation-symmetric and attention cannot tell the stem tip from a corner.
        self.point_emb = nn.Parameter(torch.randn(1, 1, N_TEE_POINTS, self.hidden_dim) * 0.02)
        nn.init.xavier_uniform_(self.point_proj.weight)
        nn.init.zeros_(self.point_proj.bias)

    def _make_net(self) -> PhaseTransformer:
        return PhaseTransformer(
            out_dim=ACT_DIM, n_tokens=self.horizon, hidden_dim=self.hidden_dim,
            num_layers=self.num_layers, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
            dropout=self.dropout, phase_dim=self.phase_dim, use_phase=self.use_phase,
            use_self_attn=False,
        )

    @property
    def n_context_tokens(self) -> int:
        return (2 + N_TEE_POINTS) * self.obs_steps

    def tokens(self, obs: torch.Tensor) -> torch.Tensor:
        """``(B, To, 5)`` raw states -> ``(B, 12 To, hidden_dim)`` context tokens.

        The parent's block and agent tokens in slots 0 and 1, then one token per point of the T.
        """
        e = embed_state(obs, self.pos)  # (B, To, 6)
        block = self.block_proj(e[..., 2:6]).unsqueeze(2)  # (B, To, 1, hidden_dim)
        agent = self.agent_proj(e[..., 0:2]).unsqueeze(2)
        points = self.point_proj(tee_points(obs, self.pos, self.tee_local)) + self.point_emb
        tok = torch.cat([block, agent, points], dim=2) + self.step_emb  # (B, To, 12, hidden_dim)
        return tok.flatten(1, 2)


ARCHS: dict[str, type] = {
    "mlp": PushTMLP,
    "mlp_medium": PushTMediumMLP,
    "mlp_medium_deep": PushTMediumDeepMLP,
    "mlp_large": PushTLargeMLP,
    "transformer": PushTTransformer,
    "transformer_medium": PushTMediumTransformer,
    "transformer_medium_deep": PushTMediumDeepTransformer,
    "transformer_large": PushTLargeTransformer,
    "transformer_big": PushTBigTransformer,
}
_POLICIES = {cls.kind: cls for cls in ARCHS.values()}


def make_pusht_policy(arch: str, **kwargs):
    """A policy backbone by name -- mirrors :func:`fm_bc.pusht.baselines.make_pusht_baseline`."""
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {tuple(ARCHS)}, got {arch!r}")
    return ARCHS[arch](**kwargs)


def load_pusht_policy(path: str | Path, map_location="cpu"):
    """Any saved Push-T policy -- a :class:`PushTMLP` or a :class:`PushTTransformer`."""
    ckpt = torch.load(path, map_location=map_location)
    cls = _POLICIES.get(ckpt.get("kind"))
    if cls is None:
        raise ValueError(f"not a Push-T policy checkpoint: kind={ckpt.get('kind')!r} "
                         f"(expected one of {tuple(_POLICIES)})")
    model = cls(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def agent_score(
    queries: torch.Tensor,
    samples: torch.Tensor,
    bandwidth: float,
    chunk: int = 4096,
) -> torch.Tensor:
    """Gaussian-kernel ``grad log p`` at ``(Q, 2)`` queries against ``(Q, M, 2)`` per-query samples.

    Euclidean mean shift, ``sum_j w_j (x_j - q) / h^2`` with softmax kernel weights -- exactly
    ``fm_bc.divergence.score_kernel_at(queries, samples, bandwidth, metric="xy")``. Units are the
    inverse of the inputs' (1/normalized unit when both are box-normalized). Like the skate
    task's stored score, the query's own sample is not left out.
    """
    h2 = bandwidth**2
    out = []
    for lo in range(0, queries.shape[0], chunk):
        q, s = queries[lo : lo + chunk], samples[lo : lo + chunk]
        xi = s - q[:, None, :]  # (b, M, 2)
        w = torch.softmax(-(xi**2).sum(-1) / (2 * h2), dim=1)
        out.append((w[..., None] * xi).sum(1) / h2)
    return torch.cat(out)
