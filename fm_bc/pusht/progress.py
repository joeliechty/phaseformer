"""A task-progress CVAE over Push-T state windows: phase and style in one split latent.

The latent is ``z = [z_phase (scalar), z_style (Z-dim Gaussian)]`` and the KL applies **only to
``z_style``**. That asymmetry is the whole design: ``z_phase`` is the rate-free channel, so the
optimizer routes as much information through it as it can, because information in ``z_style``
costs nats. The monotonicity hinge then decides *which* information -- the temporally increasing
direction. Ground-truth phase (``PushTEpisodes.phase``) never enters the graph; it is the held-out
yardstick, and :func:`progress_rows` deliberately names it ``phase_holdout`` so it cannot be passed
to the model by accident.

**The encoder is the product.** Only :meth:`ProgressCVAE.phase` ships -- a deterministic forward
pass with no sampling and no decoder. The decoder is a training scaffold, and it is load-bearing
precisely because it is discarded: reconstruction is the only term that grounds ``z_phase`` in
task-relevant state structure, since the hinge alone is satisfied by *any* monotone-along-time
statistic, meaningful or not. If the decoder stops using ``z_phase`` (the ``phase_ablation``
diagnostic), nothing shaped the encoder and a high rank correlation means nothing.

Two facts about the margin are worth stating before anyone tunes it. With ``z_phase`` bounded to
``[0, 1]`` and a per-episode margin ``m = kappa / (L - 1)``, the constraint ``p_{t+1} - p_t >= m``
telescopes to ``p_{L-1} - p_0 >= kappa``. At ``kappa = 1`` the required rise equals the entire
available range, so the unique feasible solution is ``p_t = t / (L - 1)`` -- *exactly* the
ground-truth phase. A margin at ``kappa = 1`` is therefore phase supervision wearing a hinge, and
``kappa`` is a supervision dial rather than a free hyperparameter. And a fixed (length-independent)
margin cannot work at all here: feasibility for the 246-step episodes caps it at ``1/245``, at
which point the 30-step episodes are asked to rise by only 12 % of the range.

This module imports torch, :mod:`fm_bc.pusht.chunks` and :mod:`fm_bc.pusht.normalize` and nothing
else from the repo -- ``fm_bc.skate.policy``, ``fm_bc.skate.baselines``, ``fm_bc.skate.training``
and ``fm_bc.skate.baseline_training`` all reach ``skate_planner`` and hence Theseus, which the
Push-T subtree is deliberately free of
(``tests/pusht/test_pusht_data.py::test_pusht_imports_without_theseus``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from .chunks import OBS_DIM, embed_state, make_windows
from .constants import ROT_SCALE_PX_PER_RAD
from .data import PushTEpisodes
from .normalize import Normalizer

__all__ = ["ProgressCVAE", "progress_rows", "sample_pairs", "monotonicity_hinge",
           "load_progress_model"]

# Mirrors fm_bc.skate.policy._ACTIVATIONS. Restated rather than imported: fm_bc.skate.policy pulls
# in skate_planner -> theseus, and this subtree must import with torch alone.
_ACTIVATIONS = {"silu": nn.SiLU, "relu": nn.ReLU, "tanh": nn.Tanh, "gelu": nn.GELU}


def _mlp(dims: Sequence[int], activation: str = "silu", norm: bool = True) -> nn.Sequential:
    """``dims[0] -> ... -> dims[-1]``, activation between layers but not after the last."""
    if activation not in _ACTIVATIONS:
        raise ValueError(f"activation must be one of {sorted(_ACTIVATIONS)}, got {activation!r}")
    act = _ACTIVATIONS[activation]
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(dims) - 2:
            if norm:
                layers.append(nn.LayerNorm(b))
            layers.append(act())
    return nn.Sequential(*layers)


# -- rows and pairing --------------------------------------------------------------------


def progress_rows(
    ep: PushTEpisodes,
    to: int = 2,
    device: str | torch.device = "cpu",
    episode_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """One row per step of the selected episodes: window, conditioning and pairing indices.

    ``obs`` and the flat indices come straight from :func:`fm_bc.pusht.chunks.make_windows`, so
    windows are clipped to their own episode by the same code the action-chunk baselines use.
    ``tp=1`` there is the cheapest legal horizon; the action chunk is unused.

    ``phase_holdout`` is the ground-truth phase. It is returned for **evaluation only** and is
    named so that it cannot be mistaken for a model input -- nothing in :class:`ProgressCVAE`
    accepts it.
    """
    ep = ep.to(device)
    w = make_windows(ep, to=to, tp=1, device=device, episode_ids=episode_ids)
    obs_idx = w["obs_idx"]  # (R, To)
    # The window's last observed index IS its current step: episode_window_indices builds
    # clamp(k - To + 1 + arange(To), lo, hi) and k always lies inside its own episode, so the
    # final column is exactly k.
    step = obs_idx[:, -1]
    episode = ep.episode_index[step]

    # Scatter inverting the gather, so a partner step can be turned back into a row even when
    # episode_ids has made the rows a subsequence of the flat buffer.
    row_of_step = torch.full((len(ep),), -1, dtype=torch.long, device=step.device)
    row_of_step[step] = torch.arange(step.numel(), device=step.device)

    return {
        "obs": w["obs"],  # (R, To, 5) raw states
        "cond": ep.state[ep.episode_starts[episode]],  # (R, 5) episode-initial state
        "step": step,  # (R,)
        "episode": episode,  # (R,)
        "episode_ends": ep.episode_ends,  # (E,)
        "row_of_step": row_of_step,  # (T,)
        "length": ep.lengths[episode].to(torch.float32),  # (R,)
        "phase_holdout": ep.phase[step],  # (R,) EVALUATION ONLY
    }


def sample_pairs(
    rows: dict[str, torch.Tensor],
    idx: torch.Tensor,
    gap: str = "loguniform",
    stride: int = 1,
    kappa: float = 0.5,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Partner rows ``Delta`` steps later **in the same episode**, with validity and margin.

    Rows are emitted one per flat step in buffer order, so ``idx + 1`` is the next step *except*
    across an episode boundary -- and under ``episode_ids`` filtering it is not even the next step.
    A naive increment silently pairs one episode's last window with the next episode's first, so
    the partner is always resolved through ``row_of_step``.

    ``gap="loguniform"`` draws ``Delta`` log-uniformly in ``[1, L_e - 1]``. Adjacent-only pairs
    give ``O(L)`` constraints that a high-frequency feature can satisfy while the global ordering
    drifts; long-gap pairs have large input differences and cannot be satisfied by amplifying a
    noise direction. It costs one extra random draw.

    Returns ``(partner_row, valid, margin, delta)``, all ``(B,)``. Invalid entries have
    ``partner_row = 0`` as a sentinel -- they are still encoded, because a fixed-shape batch is
    cheaper than a dynamic one at this size, and are masked out of the hinge's mean.
    """
    if gap not in ("loguniform", "fixed"):
        raise ValueError(f"gap must be 'loguniform' or 'fixed', got {gap!r}")
    length = rows["length"][idx]  # (B,)
    span = (length - 1.0).clamp(min=1.0)
    if gap == "loguniform":
        u = torch.rand(idx.shape, generator=generator, device=idx.device, dtype=span.dtype)
        delta = torch.exp(u * torch.log(span)).round().clamp(min=1.0)
    else:
        delta = torch.full_like(span, float(stride))

    step = rows["step"][idx]
    next_step = step + delta.long()
    # episode_ends is the same exclusive bound episode_window_indices clamps to. A pair reaching
    # past it has no successor, and pairing it with the clamped final step would feed the hinge a
    # zero increment it can never satisfy.
    valid = next_step < rows["episode_ends"][rows["episode"][idx]]
    partner = rows["row_of_step"][next_step.clamp(max=rows["row_of_step"].numel() - 1)]
    valid = valid & (partner >= 0)
    # Length-normalized: the only form that is feasible for L = 246 and still tight for L = 30.
    margin = kappa * delta / span
    return partner.clamp(min=0), valid, margin, delta


def monotonicity_hinge(
    z_cur: torch.Tensor,
    z_next: torch.Tensor,
    valid: torch.Tensor,
    margin: torch.Tensor,
    kappa_hi: float | None = None,
    kappa: float = 0.5,
    upper_weight: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean relative rate deficit over valid pairs, and the per-pair advance.

    ``relu(margin - advance) / margin`` is dimensionless and independent of episode length, so the
    loss reads directly as "mean fractional rate deficit" in ``[0, 1]`` and one weight means the
    same thing for a 30-step and a 246-step episode. The mean is over *valid* pairs rather than
    over the batch, so the term's scale does not drift with how many episode-final rows a batch
    happened to draw.

    ``kappa_hi`` adds the upper branch of a two-sided band, bounding the advance rate from above so
    a step-function code (all the rise in a few frames) cannot satisfy the constraint. Off by
    default.
    """
    advance = z_next - z_cur  # (B,)
    weight = valid.to(advance.dtype)
    n = weight.sum().clamp(min=1.0)
    penalty = torch.relu(margin - advance) / margin.clamp(min=1e-8)
    if kappa_hi is not None:
        rate_cap = margin * (kappa_hi / max(kappa, 1e-8))
        penalty = penalty + upper_weight * torch.relu(advance - rate_cap) / rate_cap.clamp(min=1e-8)
    return (penalty * weight).sum() / n, advance


# -- model -------------------------------------------------------------------------------


class ProgressCVAE(nn.Module):
    """``(state window (B, To, 5), episode-initial state (B, 5)) -> z_phase, z_style``.

    The shipped inference path is :meth:`phase`; :meth:`decode` exists to train the encoder.
    """

    kind: str = "progress_cvae"

    def __init__(
        self,
        pos_norm: Normalizer | dict,
        obs_steps: int = 2,
        z_style: int = 8,
        hidden: Sequence[int] = (256, 256),
        activation: str = "silu",
        beta: float = 5e-3,
        free_bits: float = 0.03,
        mono_weight: float = 1.0,
        kappa: float = 0.5,
        kappa_hi: float | None = None,
        sat_weight: float = 0.01,
        sat_limit: float = 4.0,
        norm_weight: float = 0.01,
        style_dropout: float = 0.0,
        logvar_clamp: float = 8.0,
    ):
        super().__init__()
        if isinstance(pos_norm, dict):
            pos_norm = Normalizer.from_dict(pos_norm)
        scale = torch.as_tensor(pos_norm.scale, dtype=torch.float32).flatten()
        offset = torch.as_tensor(pos_norm.offset, dtype=torch.float32).flatten()
        if scale.numel() != 2:
            raise ValueError(
                "pos_norm must be the 2-D agent_pos normalizer (embed_state applies it "
                f"separately to agent xy and block xy), got {scale.numel()} dimensions"
            )
        if z_style < 1:
            raise ValueError(f"z_style must be >= 1, got {z_style}")
        if obs_steps < 1:
            raise ValueError(f"obs_steps must be >= 1, got {obs_steps}")

        self._init_args = {
            "obs_steps": obs_steps, "z_style": z_style, "hidden": list(hidden),
            "activation": activation, "beta": beta, "free_bits": free_bits,
            "mono_weight": mono_weight, "kappa": kappa, "kappa_hi": kappa_hi,
            "sat_weight": sat_weight, "sat_limit": sat_limit, "norm_weight": norm_weight,
            "style_dropout": style_dropout, "logvar_clamp": logvar_clamp,
        }
        self.obs_steps, self.z_style = obs_steps, z_style
        self.beta, self.free_bits = beta, free_bits
        self.mono_weight, self.kappa, self.kappa_hi = mono_weight, kappa, kappa_hi
        self.sat_weight, self.sat_limit = sat_weight, sat_limit
        self.norm_weight, self.style_dropout = norm_weight, style_dropout
        self.logvar_clamp = logvar_clamp

        self.register_buffer("pos_scale", scale)
        self.register_buffer("pos_offset", offset)
        # Under box normalization one normalized unit is 256 px, so the cos/sin pair -- whose
        # variance is ~0.5 against ~0.09 for box-normalized positions -- would otherwise take
        # ~74 % of the reconstruction gradient purely as an accident of the embedding. Weight it
        # by the repo's committed px-per-radian constant against the actual position scale, which
        # also makes the weighting adapt to the minmax and gaussian modes.
        self.register_buffer(
            "w_theta", (ROT_SCALE_PX_PER_RAD / scale.mean()) ** 2
        )

        dims = list(hidden)
        self.trunk = _mlp([obs_steps * OBS_DIM + OBS_DIM] + dims, activation)
        self.phase_head = nn.Linear(dims[-1], 1)
        self.style_head = nn.Linear(dims[-1], 2 * z_style)
        self.decoder = _mlp([1 + z_style + OBS_DIM] + dims + [obs_steps * OBS_DIM], activation)

    @property
    def pos(self) -> Normalizer:
        return Normalizer(scale=self.pos_scale, offset=self.pos_offset)

    # -- encoder ------------------------------------------------------------------------

    def embed(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor | None = None,
        cond_keep: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``(B, To, 5)`` + ``(B, 5)`` -> ``(B, To*6 + 6)``.

        ``cond=None`` substitutes a zero conditioning block, which is also what ``cond_keep=False``
        does per row. One code path, so the training dropout and the no-conditioning inference
        default cannot drift apart.
        """
        x = embed_state(obs, self.pos).flatten(1)  # (B, To*6)
        if cond is None:
            c = x.new_zeros((x.shape[0], OBS_DIM))
        else:
            c = embed_state(cond, self.pos)
            if cond_keep is not None:
                c = c * cond_keep.to(c.dtype)[:, None]
        return torch.cat([x, c], dim=-1)

    def encode(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor | None = None,
        cond_keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """-> ``u (B,)`` pre-activation, ``z_phase (B,)`` in ``(0, 1)``, ``mu``/``logvar (B, Z)``."""
        h = self.trunk(self.embed(obs, cond, cond_keep))
        u = self.phase_head(h).squeeze(-1)
        mu, logvar = self.style_head(h).chunk(2, dim=-1)
        return u, torch.sigmoid(u), mu, logvar.clamp(-self.logvar_clamp, self.logvar_clamp)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        eps = torch.randn(mu.shape, generator=generator, device=mu.device, dtype=mu.dtype)
        return mu + eps * (0.5 * logvar).exp()

    # -- decoder (training scaffold) ----------------------------------------------------

    def decode(
        self, z_phase: torch.Tensor, z_style: torch.Tensor, cond: torch.Tensor | None = None,
        cond_keep: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """-> ``pos_hat (B, To, 4)`` and the raw heading vector ``v (B, To, 2)``.

        ``z_phase`` enters as ``2p - 1``, the phase convention of ``fm_bc.skate.policy.FlowMLP``.
        The heading is emitted as a 2-vector to be projected onto the unit circle, never as an
        angle: a raw angle reintroduces exactly the wrap discontinuity the cos/sin embedding exists
        to avoid.
        """
        if cond is None:
            c = z_style.new_zeros((z_style.shape[0], OBS_DIM))
        else:
            c = embed_state(cond, self.pos)
            if cond_keep is not None:
                c = c * cond_keep.to(c.dtype)[:, None]
        h = torch.cat([(2.0 * z_phase - 1.0)[:, None], z_style, c], dim=-1)
        out = self.decoder(h).view(-1, self.obs_steps, OBS_DIM)
        return out[..., :4], out[..., 4:6]

    def decode_state(self, *args, **kwargs) -> torch.Tensor:
        """The reconstruction in physical units, ``(B, To, 5)`` pixels and radians."""
        pos_hat, v = self.decode(*args, **kwargs)
        agent = self.pos.invert(pos_hat[..., :2])
        block = self.pos.invert(pos_hat[..., 2:4])
        unit = _unit(v)
        theta = torch.atan2(unit[..., 1], unit[..., 0])[..., None]
        return torch.cat([agent, block, theta], dim=-1)

    # -- the shipped inference path -----------------------------------------------------

    @torch.no_grad()
    def phase(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor | None = None,
        return_ood: bool = False,
    ):
        """``z_phase (B,)`` -- deterministic, no sampling, and **the decoder is never called**.

        With ``return_ood`` also returns a Mahalanobis-style novelty score ``||mu||^2 / Z``. The
        style posterior is already a trained density model of demonstration states, and the KL
        pulls its aggregate toward ``N(0, I)``, so the score is small and tightly concentrated on
        states like the demonstrations and grows under shift. Its *absolute* scale depends on how
        hard the KL bit during training, so compare against the ``ood_p95`` recorded in
        ``metrics.json`` rather than against 1. It is deliberately not reconstruction error: that
        would mean shipping the decoder.
        """
        _, z_phase, mu, _ = self.encode(obs, cond)
        if not return_ood:
            return z_phase
        return z_phase, mu.pow(2).sum(-1) / self.z_style

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        z_phase: torch.Tensor,
        n_style: int = 1,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """``(B, n_style, To, 5)``: prior style draws at a fixed progress, decoded to pixels.

        Fix ``cond`` and ``z_phase``, vary the style draw, and the spread is the multimodality --
        sweeping the block from one side against the other.
        """
        b = cond.shape[0]
        cond_r = cond.repeat_interleave(n_style, dim=0)
        phase_r = z_phase.repeat_interleave(n_style, dim=0)
        z = torch.randn((b * n_style, self.z_style), generator=generator,
                        device=cond.device, dtype=cond.dtype)
        return self.decode_state(phase_r, z, cond_r).view(b, n_style, self.obs_steps, 5)

    # -- objective ----------------------------------------------------------------------

    def recon_error(
        self, obs: torch.Tensor, pos_hat: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """``(B,)`` weighted squared error, summed over frames -- what the ablations compare.

        The regularizer is excluded on purpose: it is shared by every ablation and including it
        would only dilute the ratios.
        """
        x = embed_state(obs, self.pos)  # (B, To, 6)
        unit = _unit(v)
        pos_err = (pos_hat - x[..., :4]).pow(2).sum(-1)  # (B, To)
        # ||unit - (cos, sin)||^2 == 2 (1 - cos dtheta), which is a von Mises NLL up to constants.
        ang_err = (unit - x[..., 4:6]).pow(2).sum(-1)  # (B, To)
        return (pos_err + self.w_theta * ang_err).sum(-1)

    def reconstruction(
        self, obs: torch.Tensor, pos_hat: torch.Tensor, v: torch.Tensor
    ) -> torch.Tensor:
        """Weighted squared error, summed over frames and dimensions, meaned over the batch."""
        norm_reg = (v.norm(dim=-1) - 1.0).pow(2).sum(-1)  # keeps the projection conditioned
        return self.recon_error(obs, pos_hat, v).mean() + self.norm_weight * norm_reg.mean()

    def loss(
        self,
        obs: torch.Tensor,
        cond: torch.Tensor | None = None,
        obs_next: torch.Tensor | None = None,
        valid: torch.Tensor | None = None,
        margin: torch.Tensor | None = None,
        cond_keep: torch.Tensor | None = None,
        beta_scale: float = 1.0,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """``(total, extras)``. Note there is no ``phase`` argument -- by construction."""
        u, z_phase, mu, logvar = self.encode(obs, cond, cond_keep)
        z_style = self.reparameterize(mu, logvar, generator)
        if self.style_dropout > 0.0 and self.training:
            keep = (torch.rand(z_style.shape[:1], generator=generator, device=z_style.device)
                    >= self.style_dropout)
            z_style = z_style * keep.to(z_style.dtype)[:, None]
        pos_hat, v = self.decode(z_phase, z_style, cond, cond_keep)
        recon = self.reconstruction(obs, pos_hat, v)

        # The KL sees only the style block. That is what makes z_phase the channel the optimizer
        # prefers -- it is free -- and the hinge is what decides what it carries.
        kl_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean(0)  # (Z,)
        kl = kl_dim.clamp(min=self.free_bits).sum()

        mono = recon.new_zeros(())
        violation = float("nan")
        if obs_next is not None:
            # Same episode as obs by construction, so the same conditioning -- never re-gather.
            _, z_next, _, _ = self.encode(obs_next, cond, cond_keep)
            mono, advance = monotonicity_hinge(
                z_phase, z_next, valid, margin, self.kappa_hi, self.kappa
            )
            n_valid = valid.sum().clamp(min=1)
            violation = float(((advance <= 0) & valid).sum() / n_valid)

        sat = torch.relu(u.abs() - self.sat_limit).pow(2).mean()
        total = (recon + self.beta * beta_scale * kl
                 + self.mono_weight * mono + self.sat_weight * sat)
        extras = {
            "recon_loss": float(recon.detach()),
            "kl_loss": float(kl.detach()),
            "mono_loss": float(mono.detach()),
            "sat_loss": float(sat.detach()),
            "beta_eff": self.beta * beta_scale,
            "mono_violation_rate": violation,
            "active_units": int((mu.detach().var(0) > 0.01).sum()),
            "z_phase_mean": float(z_phase.detach().mean()),
            "z_phase_std": float(z_phase.detach().std()) if z_phase.numel() > 1 else 0.0,
            "sat_frac": float((u.detach().abs() > self.sat_limit).to(u.dtype).mean()),
            "mean_sigmoid_grad": float((z_phase.detach() * (1 - z_phase.detach())).mean()),
        }
        return total, extras

    # -- io -----------------------------------------------------------------------------

    def config(self) -> dict:
        return {
            "pos_norm": {"scale": self.pos_scale.tolist(), "offset": self.pos_offset.tolist()},
            **self._init_args,
        }

    def save(self, path: str | Path, **extra) -> None:
        torch.save({"kind": self.kind, "config": self.config(),
                    "state_dict": self.state_dict(), **extra}, path)


def _unit(v: torch.Tensor) -> torch.Tensor:
    """``v`` projected onto the unit circle, guarding the degenerate zero vector."""
    return v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def load_progress_model(
    path: str | Path, map_location: str | torch.device = "cpu"
) -> tuple[ProgressCVAE, dict]:
    """Rebuild a saved :class:`ProgressCVAE`; mirrors :func:`fm_bc.skate.baselines.load_policy`."""
    ckpt = torch.load(path, map_location=map_location)
    if ckpt.get("kind") != ProgressCVAE.kind:
        raise ValueError(f"not a {ProgressCVAE.kind} checkpoint: kind={ckpt.get('kind')!r}")
    cfg = dict(ckpt["config"])
    model = ProgressCVAE(**cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt
