"""Phase-conditioned velocity fields on SE(2), and their divergence.

The policy maps a pose ``T = [x, y, theta]`` and phase ``rho in [0, 1]`` to skate controls
``[v, omega]``, i.e. the body twist ``[v, 0, omega]``. Emitting controls rather than a free
3-vector keeps the skate constraint by construction, so a closed-loop rollout is an exact skate
trajectory -- the same thing the demonstrations are.

Two backbones do that, selected by :data:`ARCHS` / :func:`make_flow_policy` and by
``TrainConfig.arch``: :class:`FlowMLP` concatenates ``2 rho - 1`` to its input, and
:class:`FlowTransformer` injects the same ``rho`` by adaptive layer norm instead. They share a call
signature, so :func:`model_divergence`, the training loop and the rollouts take either.

**Divergence** of the field ``T -> T xi(T)^`` with respect to Haar measure is the trace of the
body-frame Jacobian, ``sum_i d/de_i xi_i(T Exp(e)) |_{e=0}``. SE(2) is unimodular, so the
left-invariant frame fields are themselves divergence-free and add no correction term. This is
the tangent space the kNN target lives in (``pairwise_twists`` measures distances with the same
body-frame logs), so the two divergences are directly comparable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Sequence

import torch
from torch import nn

from skate_planner.skate_model import LATERAL_INDEX, body_twist

from ..nets.transformer import PhaseTransformer
from ..se2 import se2_exp_retract
from .scenarios import BOUNDS

__all__ = ["ARCHS", "FlowMLP", "FlowTransformer", "make_flow_policy", "model_divergence"]

_ACTIVATIONS = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh}


class FlowMLP(nn.Module):
    """``(pose, phase) -> [v, omega]``.

    Input embedding ``[(x - cx)/hx, (y - cy)/hy, cos theta, sin theta, 2 rho - 1]`` with the
    world bounds as the position scale. SiLU is the default activation: a ReLU network's input
    Jacobian is piecewise constant, so its divergence field is discontinuous and the CDM term's
    gradient through it degenerates.

    ``use_phase=False`` drops the phase from the input: a plain state-feedback MLP
    ``pose -> [v, omega]`` (behaviour cloning with no flow-time signal). ``phase`` is still
    accepted everywhere and ignored, so rollouts and divergence need no special case.

    ``horizon > 1`` is the action-chunking variant: from ``(pose, phase)`` at step ``k`` it
    predicts the controls ``u_k .. u_{k+horizon-1}``, ``(..., horizon, 2)``, and a rollout
    executes the first ``action_steps`` of them before re-planning. Its :meth:`twist` is the
    first control's, so the divergence is that of the field ``u_0(pose, phase)``.
    """

    def __init__(self, hidden: Sequence[int] = (256, 256, 256), activation: str = "silu",
                 use_phase: bool = True, horizon: int = 1, action_steps: int = 1):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"activation must be one of {tuple(_ACTIVATIONS)}, got {activation!r}")
        if not 0 < action_steps <= horizon:
            raise ValueError(f"need 0 < action_steps <= horizon, got {action_steps}, {horizon}")
        self.hidden = tuple(int(h) for h in hidden)
        self.activation = activation
        self.use_phase = bool(use_phase)
        self.horizon, self.action_steps = int(horizon), int(action_steps)
        layers: list[nn.Module] = []
        width = 5 if self.use_phase else 4
        for h in self.hidden:
            layers += [nn.Linear(width, h), _ACTIVATIONS[activation]()]
            width = h
        layers.append(nn.Linear(width, 2 * self.horizon))
        self.net = nn.Sequential(*layers)
        x0, y0, x1, y1 = BOUNDS
        self.register_buffer("pos_center", torch.tensor([(x0 + x1) / 2, (y0 + y1) / 2]))
        self.register_buffer("pos_scale", torch.tensor([(x1 - x0) / 2, (y1 - y0) / 2]))

    def embed(self, poses: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        xy = (poses[..., :2] - self.pos_center) / self.pos_scale
        th = poses[..., 2:3]
        feats = [xy, torch.cos(th), torch.sin(th)]
        if self.use_phase:
            feats.append(2 * phase[..., None] - 1)
        return torch.cat(feats, dim=-1)

    def forward(self, poses: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Controls ``[v, omega]``, ``(..., 2)``; a chunk ``(..., horizon, 2)`` if chunked."""
        out = self.net(self.embed(poses, phase))
        return out if self.horizon == 1 else out.unflatten(-1, (self.horizon, 2))

    def twist(self, poses: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Body twist ``[v, 0, omega]`` of the (first) control, ``(..., 3)``."""
        u = self(poses, phase)
        return body_twist(u if self.horizon == 1 else u[..., 0, :])

    def config(self) -> dict:
        return {"hidden": list(self.hidden), "activation": self.activation,
                "use_phase": self.use_phase, "horizon": self.horizon,
                "action_steps": self.action_steps}

    def save(self, path: str | Path, **extra) -> None:
        torch.save({"config": self.config(), "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> tuple["FlowMLP", dict]:
        ckpt = torch.load(path, map_location=map_location)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model, ckpt


class FlowTransformer(nn.Module):
    """``(pose, phase) -> [v, omega]``: :class:`FlowMLP`'s interface over an AdaLN transformer.

    The pose embedding is :meth:`FlowMLP.embed`'s, minus the phase -- ``[(x - cx)/hx, (y - cy)/hy,
    cos theta, sin theta]`` -- projected to a single context token, and the phase drives adaptive
    layer norm throughout (:class:`fm_bc.nets.transformer.PhaseTransformer`) rather than entering as
    an input feature. ``use_phase=False`` accepts and ignores the phase, as :class:`FlowMLP` does.

    Leading batch dimensions are flattened and restored, so ``(..., 3) -> (..., 2)`` like
    :class:`FlowMLP`, and :func:`model_divergence` works unchanged -- the attention here is written
    out rather than using ``scaled_dot_product_attention`` precisely so the CDM term's
    ``create_graph=True`` double backward is supported (see :mod:`fm_bc.nets.transformer`). That
    double backward runs through every attention layer, so a CDM run with this backbone is
    substantially more expensive than with :class:`FlowMLP`.

    **Attention only does work when there is more than one token.** The context is a single pose
    token, so cross-attention is a no-op, and at ``horizon = 1`` there is also a single query token,
    making self-attention one too -- the model is then an AdaLN-conditioned residual MLP. It is at
    ``horizon > 1`` (the ``fm_chunk`` arm) that the self-attention over the chunk earns its cost.
    """

    kind: str = "flow_transformer"
    _POSE_DIM = 4  # [x_n, y_n, cos theta, sin theta]

    def __init__(self, use_phase: bool = True, horizon: int = 1, action_steps: int = 1,
                 hidden_dim: int = 128, num_layers: int = 2, num_heads: int = 4,
                 mlp_ratio: float = 4.0, dropout: float = 0.0, phase_dim: int = 128):
        super().__init__()
        if not 0 < action_steps <= horizon:
            raise ValueError(f"need 0 < action_steps <= horizon, got {action_steps}, {horizon}")
        self.use_phase = bool(use_phase)
        self.horizon, self.action_steps = int(horizon), int(action_steps)
        self.hidden_dim, self.num_layers = int(hidden_dim), int(num_layers)
        self.num_heads, self.mlp_ratio = int(num_heads), float(mlp_ratio)
        self.dropout, self.phase_dim = float(dropout), int(phase_dim)
        self.pose_proj = nn.Linear(self._POSE_DIM, self.hidden_dim)
        nn.init.xavier_uniform_(self.pose_proj.weight)
        nn.init.zeros_(self.pose_proj.bias)
        self.net = PhaseTransformer(
            out_dim=2, n_tokens=self.horizon, hidden_dim=self.hidden_dim,
            num_layers=self.num_layers, num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
            dropout=self.dropout, phase_dim=self.phase_dim, use_phase=self.use_phase,
        )
        x0, y0, x1, y1 = BOUNDS
        self.register_buffer("pos_center", torch.tensor([(x0 + x1) / 2, (y0 + y1) / 2]))
        self.register_buffer("pos_scale", torch.tensor([(x1 - x0) / 2, (y1 - y0) / 2]))

    def embed(self, poses: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """``(..., 3) -> (..., 4)``; ``phase`` is accepted for signature parity and unused (it
        reaches the network through the modulation, not the input)."""
        xy = (poses[..., :2] - self.pos_center) / self.pos_scale
        th = poses[..., 2:3]
        return torch.cat([xy, torch.cos(th), torch.sin(th)], dim=-1)

    def forward(self, poses: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Controls ``[v, omega]``, ``(..., 2)``; a chunk ``(..., horizon, 2)`` if chunked."""
        lead = poses.shape[:-1]
        context = self.pose_proj(self.embed(poses, phase)).reshape(-1, 1, self.hidden_dim)
        u = self.net(context, phase.reshape(-1).to(context))
        return u[:, 0].reshape(*lead, 2) if self.horizon == 1 else u.reshape(*lead, self.horizon, 2)

    def twist(self, poses: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Body twist ``[v, 0, omega]`` of the (first) control, ``(..., 3)``."""
        u = self(poses, phase)
        return body_twist(u if self.horizon == 1 else u[..., 0, :])

    def config(self) -> dict:
        return {"use_phase": self.use_phase, "horizon": self.horizon,
                "action_steps": self.action_steps, "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers, "num_heads": self.num_heads,
                "mlp_ratio": self.mlp_ratio, "dropout": self.dropout, "phase_dim": self.phase_dim}

    def save(self, path: str | Path, **extra) -> None:
        torch.save({"kind": self.kind, "config": self.config(),
                    "state_dict": self.state_dict(), **extra}, path)

    @classmethod
    def load(cls, path: str | Path, map_location="cpu") -> tuple["FlowTransformer", dict]:
        ckpt = torch.load(path, map_location=map_location)
        if ckpt.get("kind") != cls.kind:
            raise ValueError(f"not a {cls.kind} checkpoint: kind={ckpt.get('kind')!r}")
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model, ckpt


ARCHS: dict[str, type] = {"mlp": FlowMLP, "transformer": FlowTransformer}


def make_flow_policy(arch: str, use_phase: bool = True, horizon: int = 1, action_steps: int = 1,
                     hidden=(256, 256, 256), activation: str = "silu", **net_kwargs):
    """A skate policy backbone by name; ``hidden``/``activation`` apply to ``"mlp"`` only."""
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {tuple(ARCHS)}, got {arch!r}")
    if arch == "mlp":
        if net_kwargs:
            raise ValueError(f"unexpected kwargs for arch='mlp': {sorted(net_kwargs)}")
        return FlowMLP(hidden, activation, use_phase, horizon, action_steps)
    return FlowTransformer(use_phase=use_phase, horizon=horizon, action_steps=action_steps,
                           **net_kwargs)


def model_divergence(
    model: "FlowMLP | FlowTransformer",
    poses: torch.Tensor,
    phase: torch.Tensor,
    estimator: Literal["exact", "hutchinson"] = "exact",
    n_probes: int = 1,
    create_graph: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predicted twist ``(B, 3)`` and its SE(2) divergence ``(B,)`` in 1/s.

    One forward pass at ``T Exp(e)``, ``e = 0``, so the prediction and its divergence share a
    graph and the twist can go straight into the flow-matching loss. ``create_graph=True`` keeps
    the divergence differentiable in the parameters (needed for the CDM loss); the forward graph
    is always retained so the caller can still backpropagate through the twist.

    ``"exact"`` takes the trace one diagonal entry at a time (D = 3, so this is cheap and has
    no estimator noise); the lateral row is skipped because a skate policy's lateral twist is
    identically zero. What remains, ``dv/de_fwd + domega/de_yaw``, is
    ``cos(theta) dv/dx + sin(theta) dv/dy + domega/dtheta``: only the directions a skate can
    drive.

    ``"hutchinson"`` is ``E[z^T J z]`` with ``z ~ N(0, diag(1, 0, 1))``: Gaussian probes over
    the same two admissible body directions, pushed through ``se2_exp_retract`` like any other
    body twist. The lateral row of ``J`` is zero, so an isotropic probe (the reference's
    ``N(0, I)``) has the same expectation but adds the cross terms ``dv/de_lat`` and
    ``domega/de_lat`` -- variance, never trace.
    """
    with torch.enable_grad():
        eps = torch.zeros_like(poses, requires_grad=True)
        twist = model.twist(se2_exp_retract(poses, eps), phase)
        if estimator == "exact":
            div = poses.new_zeros(poses.shape[0])
            for i in range(3):
                if i == LATERAL_INDEX:
                    continue
                (g,) = torch.autograd.grad(
                    twist[:, i].sum(), eps, create_graph=create_graph, retain_graph=True
                )
                div = div + g[:, i]
        elif estimator == "hutchinson":
            div = poses.new_zeros(poses.shape[0])
            for _ in range(n_probes):
                z = torch.randn_like(poses)
                z[:, LATERAL_INDEX] = 0.0
                (g,) = torch.autograd.grad(
                    (twist * z).sum(), eps, create_graph=create_graph, retain_graph=True
                )
                div = div + (g * z).sum(dim=-1)
            div = div / n_probes
        else:
            raise ValueError(f"estimator must be 'exact' or 'hutchinson', got {estimator!r}")
    return twist, div
