"""A supervised MLP that estimates the phase from the state, and its monotone rollout wrapper.

The nominal clock (:class:`fm_bc.pusht.sim.NominalClock`) advances at a fixed rate and is blind to
whether the block is ahead of or behind schedule: at ``rho = 1`` its rollouts have made only
0.50-0.74 of the geometric progress a demonstration has made by then. This module replaces the
clock with a regressor of the oracle phase.

**Model.** A three-layer MLP on the current state -- agent ``xy``, block ``xy`` and ``cos``/``sin``
of the block heading, all box-normalized -- with a sigmoid output, trained by MSE against the
demonstrations' ``rho = k / (L - 1)``. With ``use_start`` the episode's *initial* block pose is
appended, so the network can read progress as a displacement from the start rather than as an
absolute pose: without it the estimate at a rollout's first step averages 0.10 instead of 0,
because a start state the demonstrations never begin from looks mid-episode. It is deliberately allowed to overfit: it is an instrument
for injecting task progress into the policy, not a generalization claim.

**Step size.** ``step`` decouples the two jobs the phase does. Left at ``None`` the increment that
scales the rollout step is the estimate's own, ``drho = rho_t - rho_{t-1}``, which is the paper's
integration but deadlocks: a state-based phase only advances once the block moves, and the block
only moves if ``drho`` is large enough to move the pusher. With ``step = 1 / (L_med - 1)`` the
policy is *conditioned* on the estimate while the step stays the median demonstration's.

**Monotone wrapper.** A regressor is not monotone and can stand still, which would break
``u = dx/drho``. :class:`MonotonePhase` bounds it below,

    rho_t = max(rho_{t-1} + eps, MLP(state_t)),    eps = 1 / 200 by default,

so the phase never moves backwards and ``drho >= eps > 0`` -- the robot can neither freeze nor
divide by zero. The phase *fed to the policy* is clipped to 1 (the training range); the increment
that scales the step, ``drho = rho_t - rho_{t-1}``, is the unclipped one, so a saturated estimate
still steps at ``eps``.

``ratchet=False`` drops the bound -- ``rho_t = MLP(state_t)``, free to stand still or move
backwards -- the ablation of the wrapper. It needs the fixed ``step``: the estimate's own increment
can then be zero or negative, which would freeze the pusher or back it up.

**Transformer clock.** :class:`PhaseClockTransformer` reads the same inputs as tokens -- the robot,
the object now and the object at the start -- and lets them attend to each other, so progress can
be read off the object's pose against its start directly. Same call signature, same output head,
same checkpoint contract: :func:`load_phase_mlp` loads either by its ``kind``, and
:class:`MonotonePhase` wraps either.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .chunks import embed_state
from .normalize import Normalizer

__all__ = ["PhaseMLP", "PhaseClockTransformer", "MonotonePhase", "load_phase_mlp", "PHASE_EPS"]

PHASE_EPS = 1.0 / 200.0
_STATE_DIM = 6  # [agent_xy, block_xy, cos, sin]
_BLOCK_DIM = 4  # the block's [xy, cos, sin] of the start state


class PhaseMLP(nn.Module):
    """``state (B, 5) or (B, To, 5) -> rho_hat (B,) in (0, 1)``."""

    kind: str = "pusht_phase_mlp"

    def __init__(self, pos_norm: Normalizer | dict, hidden: Sequence[int] = (512, 512),
                 use_start: bool = False):
        super().__init__()
        if isinstance(pos_norm, dict):
            pos_norm = Normalizer.from_dict(pos_norm)
        scale = torch.as_tensor(pos_norm.scale, dtype=torch.float32).flatten()
        offset = torch.as_tensor(pos_norm.offset, dtype=torch.float32).flatten()
        if scale.numel() != 2:
            raise ValueError(f"pos_norm must be the 2-D agent_pos normalizer, got {scale.numel()}-D")
        self.hidden, self.use_start = tuple(int(h) for h in hidden), bool(use_start)
        self.register_buffer("pos_scale", scale)
        self.register_buffer("pos_offset", offset)
        layers: list[nn.Module] = []
        width = _STATE_DIM + _BLOCK_DIM * int(self.use_start)
        for h in self.hidden:
            layers += [nn.Linear(width, h), nn.ReLU()]
            width = h
        layers += [nn.Linear(width, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    @property
    def pos(self) -> Normalizer:
        return Normalizer(scale=self.pos_scale, offset=self.pos_offset)

    def forward(self, state: torch.Tensor, start: torch.Tensor | None = None) -> torch.Tensor:
        """``rho_hat`` of each state; ``start`` is its episode's first state (needed by
        ``use_start``, ignored otherwise)."""
        if state.dim() == 3:  # a window: the current step only
            state = state[:, -1]
        x = embed_state(state, self.pos)
        if self.use_start:
            if start is None:
                raise ValueError("this estimator needs the episode's start state")
            if start.dim() == 3:
                start = start[:, -1]
            x = torch.cat([x, embed_state(start.expand_as(state), self.pos)[..., 2:6]], dim=-1)
        return self.net(x).squeeze(-1)

    def config(self) -> dict:
        return {"pos_norm": {"scale": self.pos_scale.tolist(), "offset": self.pos_offset.tolist()},
                "hidden": list(self.hidden), "use_start": self.use_start}

    def save(self, path: str | Path, **extra) -> None:
        torch.save({"kind": self.kind, "config": self.config(),
                    "state_dict": self.state_dict(), **extra}, path)


class PhaseClockTransformer(nn.Module):
    """``state (B, 5) or (B, To, 5) -> rho_hat (B,) in (0, 1)``, with the robot and the object as
    separate tokens.

    Four tokens: a learned readout token, the robot (agent ``xy``), the object now (block ``xy``,
    ``cos``/``sin`` of its heading) and, with ``use_start``, the object at the episode's start --
    all box-normalized as :class:`PhaseMLP`'s inputs are. The two object tokens share one
    projection, so a learned per-slot embedding is what tells *now* from *start*, as
    :class:`fm_bc.pusht.policy.PushTTransformer` shares ``block_proj`` across observed steps and
    adds a step embedding. A pre-LN self-attention encoder lets every token attend to every other,
    and ``rho`` is read off the readout token through :class:`PhaseMLP`'s head, a linear layer and
    a sigmoid.

    Self-attention rather than :class:`fm_bc.nets.transformer.PhaseTransformer`, which only lets a
    learned query cross-attend to the inputs: here the object's current pose can attend to its
    start pose, which is the comparison progress is read from. And the clock has no phase input,
    so ``PhaseTransformer``'s adaLN modulations would be dead weights in the parameter count.

    The defaults (``hidden_dim=104, num_layers=2``, 263,849 parameters with ``use_start``) are
    sized to the paper's 512 x 2 MLP clock (268,801).
    """

    kind: str = "pusht_phase_transformer"
    _AGENT_DIM, _OBJECT_DIM = 2, 4  # [xy] and [xy, cos, sin] of the 6-D embedded state

    def __init__(self, pos_norm: Normalizer | dict, hidden_dim: int = 104, num_layers: int = 2,
                 num_heads: int = 4, mlp_ratio: float = 4.0, use_start: bool = False):
        super().__init__()
        if isinstance(pos_norm, dict):
            pos_norm = Normalizer.from_dict(pos_norm)
        scale = torch.as_tensor(pos_norm.scale, dtype=torch.float32).flatten()
        offset = torch.as_tensor(pos_norm.offset, dtype=torch.float32).flatten()
        if scale.numel() != 2:
            raise ValueError(f"pos_norm must be the 2-D agent_pos normalizer, got {scale.numel()}-D")
        self.hidden_dim, self.num_layers = int(hidden_dim), int(num_layers)
        self.num_heads, self.mlp_ratio = int(num_heads), float(mlp_ratio)
        self.use_start = bool(use_start)
        self.register_buffer("pos_scale", scale)
        self.register_buffer("pos_offset", offset)
        d = self.hidden_dim
        self.robot_proj = nn.Linear(self._AGENT_DIM, d)
        self.object_proj = nn.Linear(self._OBJECT_DIM, d)  # now and start alike
        # Slots: readout, robot, object now[, object at the start].
        self.slot_emb = nn.Parameter(torch.randn(1, 3 + int(self.use_start), d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, self.num_heads, int(d * self.mlp_ratio),
                                           dropout=0.0, activation="gelu", batch_first=True,
                                           norm_first=True)
        # No nested-tensor fast path: there is no padding to skip, and it warns under norm_first.
        self.encoder = nn.TransformerEncoder(layer, self.num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, 1), nn.Sigmoid())

    @property
    def pos(self) -> Normalizer:
        return Normalizer(scale=self.pos_scale, offset=self.pos_offset)

    def tokens(self, state: torch.Tensor, start: torch.Tensor | None = None) -> torch.Tensor:
        """``(B, 5)`` raw states (and their episodes' starts) -> ``(B, 3 or 4, hidden_dim)``."""
        e = embed_state(state, self.pos)  # (B, 6) [agent_xy, block_xy, cos, sin]
        parts = [self.robot_proj(e[:, None, 0:2]), self.object_proj(e[:, None, 2:6])]
        if self.use_start:
            if start is None:
                raise ValueError("this estimator needs the episode's start state")
            if start.dim() == 3:
                start = start[:, -1]
            s = embed_state(start.expand_as(state), self.pos)
            parts.append(self.object_proj(s[:, None, 2:6]))
        readout = torch.zeros_like(parts[0])  # the readout token is its slot embedding alone
        return torch.cat([readout, *parts], dim=1) + self.slot_emb

    def forward(self, state: torch.Tensor, start: torch.Tensor | None = None) -> torch.Tensor:
        """``rho_hat`` of each state -- :meth:`PhaseMLP.forward`'s contract."""
        if state.dim() == 3:  # a window: the current step only
            state = state[:, -1]
        h = self.encoder(self.tokens(state, start))
        return self.head(self.norm(h[:, 0])).squeeze(-1)

    def config(self) -> dict:
        return {"pos_norm": {"scale": self.pos_scale.tolist(), "offset": self.pos_offset.tolist()},
                "hidden_dim": self.hidden_dim, "num_layers": self.num_layers,
                "num_heads": self.num_heads, "mlp_ratio": self.mlp_ratio,
                "use_start": self.use_start}

    def save(self, path: str | Path, **extra) -> None:
        torch.save({"kind": self.kind, "config": self.config(),
                    "state_dict": self.state_dict(), **extra}, path)


_CLOCKS = {cls.kind: cls for cls in (PhaseMLP, PhaseClockTransformer)}


def load_phase_mlp(path: str | Path, map_location="cpu") -> tuple[nn.Module, dict]:
    """Any saved phase estimator -- a :class:`PhaseMLP` or a :class:`PhaseClockTransformer`, by
    the checkpoint's ``kind``."""
    ckpt = torch.load(path, map_location=map_location)
    cls = _CLOCKS.get(ckpt.get("kind"))
    if cls is None:
        raise ValueError(f"not a phase-estimator checkpoint: kind={ckpt.get('kind')!r} "
                         f"(expected one of {tuple(_CLOCKS)})")
    model = cls(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {k: v for k, v in ckpt.items() if k not in ("state_dict", "config")}


class MonotonePhase:
    """State-based phase clock: ``rho_t = max(rho_{t-1} + eps, MLP(state_t))`` (module docstring),
    or the estimate alone with ``ratchet=False``.

    Interface-compatible with :class:`fm_bc.pusht.sim.NominalClock`: ``reset()`` at the start of an
    episode, then ``advance(t, state)`` returns ``(rho fed to the policy, drho)``.
    """

    def __init__(self, model: nn.Module, eps: float = PHASE_EPS, smooth: float = 1.0,
                 max_rate: float | None = None, step: float | None = None,
                 ratchet: bool = True, device: str | torch.device = "cpu"):
        if not ratchet and step is None:
            raise ValueError("without the ratchet the estimate's own increment can be zero or "
                             "negative, freezing the pusher or backing it up: pass a fixed step")
        self.model, self.device = model.to(device).eval(), device
        self.eps, self.smooth, self.max_rate, self.step = float(eps), float(smooth), max_rate, step
        self.ratchet = ratchet
        self.rho, self.estimate, self.start = 0.0, 0.0, None

    def reset(self) -> None:
        self.rho, self.estimate, self.start = 0.0, 0.0, None

    @torch.no_grad()
    def advance(self, t: int, state) -> tuple[float, float]:
        """``(rho fed to the policy, drho)``: the ratchet above, on a smoothed, rate-capped estimate.

        ``smooth < 1`` low-passes the estimator (EMA) before the ratchet: the raw estimate is noisy,
        and ``max`` over a noisy signal latches onto its spikes, after which every later estimate
        falls below the latch and the clock degenerates to ``eps`` per step. ``max_rate`` caps how
        far one step may advance, so a jump in the estimate cannot teleport the phase.
        """
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).reshape(1, -1)
        if t == 0 or self.start is None:
            self.start = s.clone()
        raw = float(self.model(s, self.start)[0])
        self.estimate = raw if t == 0 else (1 - self.smooth) * self.estimate + self.smooth * raw
        previous = self.rho
        self.rho = (self.estimate if t == 0 or not self.ratchet
                    else max(previous + self.eps, self.estimate))
        if self.max_rate is not None and t > 0:
            self.rho = min(self.rho, previous + self.max_rate)
        return min(self.rho, 1.0), self.rho - previous if self.step is None else self.step

    @property
    def describe(self) -> dict:
        return {"clock": "phase_mlp", "eps": self.eps, "smooth": self.smooth,
                "max_rate": self.max_rate, "step": self.step, "ratchet": self.ratchet}


def train_phase_mlp(
    ep,
    prep,
    episode_ids=None,
    hidden: Sequence[int] = (512, 512),
    use_start: bool = False,
    num_epochs: int = 1500,
    batch_size: int = 1024,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    seed: int = 0,
    device: str | torch.device = "cpu",
    verbose: bool = False,
    arch: str = "mlp",
    net_kwargs: dict | None = None,
    warmup_steps: int = 0,
    cosine: bool = False,
) -> tuple[nn.Module, dict]:
    """Fit a phase estimator by MSE on ``rho = k / (L - 1)``; returns the model and its metrics.

    ``episode_ids`` defaults to *every* episode -- this estimator is meant to fit the
    demonstrations closely, so it is trained on all of them and reported per split for information
    (the per-split RMSEs are diagnostics, not a held-out claim).

    ``arch="mlp"`` is :class:`PhaseMLP` of widths ``hidden``; ``arch="transformer"`` is
    :class:`PhaseClockTransformer` built from ``net_kwargs`` (``hidden`` is then unused).
    ``warmup_steps`` ramps the learning rate up linearly and ``cosine`` anneals it to 0 over the
    run; both off is the constant rate every MLP clock was trained with.
    """
    torch.manual_seed(seed)
    ids = torch.arange(ep.n_episodes) if episode_ids is None else torch.as_tensor(episode_ids)
    rows = torch.cat([torch.arange(ep.episode_slice(int(e)).start, ep.episode_slice(int(e)).stop)
                      for e in ids])
    state = ep.state[rows].to(device=device, dtype=torch.float32)
    phase = ep.phase[rows].to(device=device, dtype=torch.float32)
    starts = torch.cat([ep.state[ep.episode_slice(int(e)).start][None].expand(
        ep.episode_slice(int(e)).stop - ep.episode_slice(int(e)).start, -1) for e in ids])
    starts = starts.to(device=device, dtype=torch.float32)
    pos = prep.normalizer("agent_pos")
    if arch == "mlp":
        model = PhaseMLP(pos, hidden, use_start=use_start)
    elif arch == "transformer":
        model = PhaseClockTransformer(pos, use_start=use_start, **(net_kwargs or {}))
    else:
        raise ValueError(f"arch must be 'mlp' or 'transformer', got {arch!r}")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n = len(rows)
    total_steps = num_epochs * -(-n // batch_size)

    def lr_factor(step: int) -> float:
        warm = min(1.0, (step + 1) / warmup_steps) if warmup_steps else 1.0
        if not cosine:
            return warm
        return warm * 0.5 * (1.0 + math.cos(math.pi * step / total_steps))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_factor)
    for epoch in range(num_epochs):
        perm = torch.randperm(n, generator=generator).to(device)
        total = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            loss = torch.nn.functional.mse_loss(model(state[idx], starts[idx]), phase[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss) * len(idx)
        if verbose and (epoch % 25 == 0 or epoch == num_epochs - 1):
            print(f"  epoch {epoch:4d}  mse {total / n:.5f}", flush=True)
    model.eval()
    with torch.no_grad():
        error = (model(state, starts) - phase).cpu()
    metrics = {"rmse": float(error.pow(2).mean().sqrt()), "mae": float(error.abs().mean()),
               "n_rows": int(n), "n_episodes": int(len(ids)), "seed": seed,
               "arch": arch, "n_params": sum(p.numel() for p in model.parameters()),
               "use_start": use_start, "num_epochs": num_epochs, "lr": lr,
               "warmup_steps": warmup_steps, "cosine": cosine}
    if arch == "mlp":
        metrics["hidden"] = list(hidden)
    else:
        metrics["net"] = {k: v for k, v in model.config().items() if k != "pos_norm"}
    episode_of = ep.episode_index[rows].cpu()
    for name, split_ids in (("train", prep.train_ids), ("val", prep.val_ids),
                            ("test", prep.test_ids)):
        mask = torch.isin(episode_of, split_ids.cpu())
        if mask.any():
            metrics[f"rmse_{name}"] = float(error[mask].pow(2).mean().sqrt())
    return model, metrics
