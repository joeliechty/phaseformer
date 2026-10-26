"""Train the Push-T progress CVAE, and score the encoder that comes out of it.

Split from :mod:`fm_bc.pusht.progress` the way :mod:`fm_bc.skate.baseline_training` is split from
:mod:`fm_bc.skate.baselines`: the model file stays free of ``json`` / ``csv`` / ``time`` and of the
filesystem, which is what makes it cheap to unit-test.

The evaluation here scores the **encoder**, because the encoder is what ships. Two consequences
run through every metric below:

* ``phase_ablation`` is a **gate**, not a diagnostic. Reconstruction is the only term that grounds
  ``z_phase`` in task-relevant structure -- the hinge alone is satisfied by any monotone-along-time
  statistic. If the decoder was not using ``z_phase``, nothing shaped the encoder and the rank
  correlations below are meaningless however good they look.
* ``spearman_pooled`` and ``calibration_gap`` are headline numbers. Per-episode Spearman is
  invariant to *any* monotone reparameterization, so it cannot see whether ``z_phase = 0.6`` means
  the same thing in two episodes -- which is exactly what reading the scalar out at inference
  assumes.

Everything is torch: no scipy (Spearman is implemented here with tie-aware average ranks) and no
import that reaches Theseus, so this module runs wherever the rest of the Push-T subtree does.
``lr_factor``, the power EMA and ``_write_log`` are therefore restated rather than imported --
and ``fm_bc.skate.training._write_log`` could not be reused in any case, because it writes against a
module-global ``_LOG_FIELDS`` with ``extrasaction="ignore"`` and would silently drop every column
this model logs.
"""

from __future__ import annotations

import copy
import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from ..se2 import se2_log, twist_norm
from .chunks import embed_state
from .constants import GOAL_POSE, ROT_SCALE_PX_PER_RAD
from .data import PushTEpisodes
from .prepared import PushTPrepared
from .progress import ProgressCVAE, progress_rows, sample_pairs

__all__ = ["ProgressConfig", "progress_run_dir", "lr_factor", "kl_scale", "ema_decay",
           "spearman", "encode_rows", "evaluate_progress", "phase_baselines", "train_progress"]

_LOG_FIELDS = [
    "step", "seconds", "lr", "beta_eff", "total_loss", "recon_loss", "kl_loss", "mono_loss",
    "sat_loss", "mono_violation_rate", "active_units", "z_phase_mean", "z_phase_std",
    "sat_frac", "mean_sigmoid_grad", "val_spearman", "val_violation_local", "val_phase_ablation",
    "val_recon",
]


@dataclass
class ProgressConfig:
    # -- data
    obs_steps: int = 2
    normalizer: str = "box"
    gap: str = "loguniform"  # "loguniform" or "fixed"
    stride: int = 1  # only used when gap == "fixed"
    kappa: float = 0.5  # margin fraction; kappa = 1 IS exact phase supervision, never run it
    kappa_hi: float | None = None  # upper band; turn on if calibration_gap is large
    cond_dropout: float = 0.1  # so the encoder works without the episode-initial state
    # -- model
    z_style: int = 8
    hidden: tuple = (256, 256)
    activation: str = "silu"
    beta: float = 5e-3  # = 2 * sigma_r^2 with sigma_r = 0.05 normalized (12.8 px)
    free_bits: float = 0.03
    mono_weight: float = 1.0
    sat_weight: float = 0.01
    style_dropout: float = 0.0
    # -- optimization
    steps: int = 20000
    batch_size: int = 256
    lr: float = 1e-3  # not the baselines' 1e-4: that is tuned for a UNet over 100k steps
    weight_decay: float = 1e-4  # 100x the baselines': only 90 training episodes here
    betas: tuple = (0.9, 0.999)
    warmup_steps: int = 500
    lr_schedule: str = "cosine"
    kl_warmup: tuple = (500, 3000)  # beta is 0 until the first, full from the second
    grad_clip: float | None = 1.0
    ema: bool = True
    ema_power: float = 0.75
    ema_max: float = 0.9999
    # -- bookkeeping
    log_every: int = 100
    eval_every: int = 1000
    seed: int = 0
    device: str = "cpu"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.lr_schedule not in ("cosine", "constant"):
            raise ValueError(f"lr_schedule must be cosine or constant, got {self.lr_schedule!r}")
        if self.gap not in ("loguniform", "fixed"):
            raise ValueError(f"gap must be loguniform or fixed, got {self.gap!r}")
        if not 0.0 < self.kappa <= 1.0:
            raise ValueError(f"kappa must be in (0, 1], got {self.kappa}")
        if not 0.0 <= self.cond_dropout < 1.0:
            raise ValueError(f"cond_dropout must be in [0, 1), got {self.cond_dropout}")
        self.betas = tuple(self.betas)
        self.hidden = tuple(self.hidden)
        self.kl_warmup = tuple(self.kl_warmup)


def progress_run_dir(root: str | Path, cfg: ProgressConfig) -> Path:
    return Path(root) / "pusht" / "progress" / f"z{cfg.z_style}_k{cfg.kappa:g}" / f"seed{cfg.seed}"


def lr_factor(cfg: ProgressConfig, step: int) -> float:
    """Linear warmup then a half cosine to 0 -- diffusers' ``get_cosine_schedule_with_warmup``.

    A verbatim restatement of :func:`fm_bc.skate.baseline_training.lr_factor`; importing it would
    pull in ``ChunkPolicy`` -> ``skate_planner`` -> theseus.
    """
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    if cfg.lr_schedule == "constant":
        return 1.0
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def kl_scale(cfg: ProgressConfig, step: int) -> float:
    """0 for the first ``kl_warmup[0]`` steps, then linear to 1 by ``kl_warmup[1]``.

    Unlike the KL, the monotonicity term is on from step 0: delayed, ``z_phase`` gets co-opted by
    reconstruction as one more style dimension in the first thousand steps and never recovers.
    """
    lo, hi = cfg.kl_warmup
    if step < lo:
        return 0.0
    return min(1.0, (step - lo) / max(1, hi - lo))


def ema_decay(cfg: ProgressConfig, step: int) -> float:
    """Diffusion Policy's power schedule; restated for the same theseus reason as ``lr_factor``."""
    s = step - 1
    if s <= 0:
        return 0.0
    return min(max(1 - (1 + s) ** -cfg.ema_power, 0.0), cfg.ema_max)


@torch.no_grad()
def _ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float) -> None:
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p.detach(), alpha=1 - decay)
    for b_ema, b in zip(ema.buffers(), model.buffers()):
        b_ema.copy_(b)


def _write_log(path: Path, log: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(log)


# -- rank correlation --------------------------------------------------------------------


def _average_ranks(x: torch.Tensor) -> torch.Tensor:
    """Ranks of ``x`` with ties averaged, ``(N,)`` float64."""
    n = x.shape[0]
    order = torch.argsort(x)
    sorted_x = x[order]
    positions = torch.arange(n, dtype=torch.float64, device=x.device)
    fresh = torch.ones(n, dtype=torch.bool, device=x.device)
    fresh[1:] = sorted_x[1:] != sorted_x[:-1]
    group = torch.cumsum(fresh.long(), 0) - 1
    n_groups = int(group[-1]) + 1
    sums = torch.zeros(n_groups, dtype=torch.float64, device=x.device).index_add_(0, group, positions)
    counts = torch.zeros(n_groups, dtype=torch.float64, device=x.device).index_add_(
        0, group, torch.ones_like(positions)
    )
    out = torch.empty(n, dtype=torch.float64, device=x.device)
    out[order] = (sums / counts)[group]
    return out


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    """Tie-aware Spearman rho, in pure torch -- ``nan`` for degenerate input.

    Ties matter: ``ep.phase`` repeats the same normalized time across episodes, so a plain
    argsort-of-argsort rank would be wrong on the pooled correlation. Implemented here rather than
    taken from scipy so the Push-T subtree keeps needing torch alone.
    """
    a, b = a.flatten().to(torch.float64), b.flatten().to(torch.float64)
    if a.numel() < 2:
        return float("nan")
    ra, rb = _average_ranks(a), _average_ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = ra.norm() * rb.norm()
    if float(denom) == 0.0:
        return float("nan")
    return float((ra * rb).sum() / denom)


def _episode_groups(rows: dict[str, torch.Tensor]) -> list[tuple[int, int, int]]:
    """``(episode, start, stop)`` row spans. Rows keep flat-buffer order, so each span is
    contiguous and already sorted by step."""
    uniq, counts = torch.unique_consecutive(rows["episode"], return_counts=True)
    spans, start = [], 0
    for e, c in zip(uniq.tolist(), counts.tolist()):
        spans.append((e, start, start + c))
        start += c
    return spans


def _percentile(values: list[float], q: float) -> float:
    clean = sorted(v for v in values if not math.isnan(v))
    if not clean:
        return float("nan")
    pos = q * (len(clean) - 1)
    lo, hi = math.floor(pos), math.ceil(pos)
    return clean[lo] + (clean[hi] - clean[lo]) * (pos - lo)


def _bootstrap_ci(values: list[float], n_boot: int = 2000, seed: int = 0) -> list[float]:
    clean = [v for v in values if not math.isnan(v)]
    if len(clean) < 2:
        return [float("nan"), float("nan")]
    g = torch.Generator().manual_seed(seed)
    t = torch.tensor(clean, dtype=torch.float64)
    idx = torch.randint(len(clean), (n_boot, len(clean)), generator=g)
    means = t[idx].mean(dim=1)
    return [float(means.quantile(0.025)), float(means.quantile(0.975))]


# -- encoding and evaluation -------------------------------------------------------------


@torch.no_grad()
def encode_rows(
    model: ProgressCVAE,
    rows: dict[str, torch.Tensor],
    cond: bool = True,
    batch_size: int = 8192,
) -> dict[str, torch.Tensor]:
    """``z_phase``, ``u``, ``mu``, ``logvar`` and ``ood`` for every row."""
    model.eval()
    out: dict[str, list[torch.Tensor]] = {k: [] for k in ("u", "z_phase", "mu", "logvar")}
    n = rows["obs"].shape[0]
    for lo in range(0, n, batch_size):
        hi = min(lo + batch_size, n)
        c = rows["cond"][lo:hi] if cond else None
        u, z, mu, logvar = model.encode(rows["obs"][lo:hi], c)
        for k, v in zip(("u", "z_phase", "mu", "logvar"), (u, z, mu, logvar)):
            out[k].append(v)
    merged = {k: torch.cat(v) for k, v in out.items()}
    merged["ood"] = merged["mu"].pow(2).sum(-1) / model.z_style
    return merged


@torch.no_grad()
def _ablations(
    model: ProgressCVAE, rows: dict[str, torch.Tensor], enc: dict[str, torch.Tensor],
    generator: torch.Generator | None = None,
) -> dict[str, float]:
    """Percent increase in reconstruction error when each decoder input is shuffled.

    ``phase_ablation`` is the gate: below ~5 % the decoder is not using ``z_phase``, so nothing in
    training shaped it and every rank correlation is an artifact.
    """
    obs, cond, z_phase, mu = rows["obs"], rows["cond"], enc["z_phase"], enc["mu"]
    n = obs.shape[0]
    perm = torch.randperm(n, generator=generator, device=obs.device)

    def err(p, z, c):
        pos_hat, v = model.decode(p, z, c)
        return float(model.recon_error(obs, pos_hat, v).mean())

    base = err(z_phase, mu, cond)
    if base <= 0:
        return {k: float("nan") for k in ("phase_ablation", "c_ablation", "style_ablation")}
    return {
        "phase_ablation": 100.0 * (err(z_phase[perm], mu, cond) - base) / base,
        "style_ablation": 100.0 * (err(z_phase, mu[perm], cond) - base) / base,
        "c_ablation": 100.0 * (err(z_phase, mu, cond[perm]) - base) / base,
    }


def _violations(
    rows: dict[str, torch.Tensor], z: torch.Tensor, kappa: float, n_long: int = 20000,
    seed: int = 0,
) -> dict[str, float]:
    """Violation rates at adjacent steps, at the training margin, and over long random gaps."""
    groups = _episode_groups(rows)
    local_bad = local_n = margin_bad = 0
    rough_num = rough_den = 0.0
    for _, lo, hi in groups:
        p = z[lo:hi]
        if p.numel() < 2:
            continue
        adv = p[1:] - p[:-1]
        local_bad += int((adv <= 0).sum())
        local_n += adv.numel()
        margin = kappa / max(p.numel() - 1, 1)
        margin_bad += int((adv < margin).sum())
        if p.numel() >= 3:
            rough_num += float((p[2:] - 2 * p[1:-1] + p[:-2]).abs().sum())
            rough_den += float(adv.abs().sum())

    g = torch.Generator(device="cpu").manual_seed(seed)
    long_bad = long_n = 0
    k = max(1, n_long // max(1, len(groups)))
    for _, lo, hi in groups:
        p = z[lo:hi].cpu()
        span = p.numel() - 1
        if span < 1:
            continue
        u = torch.rand(k, generator=g, dtype=torch.float64)
        delta = torch.exp(u * math.log(max(span, 1))).round().clamp(min=1).long()
        t = (torch.rand(k, generator=g) * (span - delta + 1).clamp(min=1)).long()
        t = t.clamp(max=span - 1)
        t2 = (t + delta).clamp(max=span)
        long_bad += int((p[t2] - p[t] <= 0).sum())
        long_n += k

    return {
        "violation_local": local_bad / max(local_n, 1),
        "violation_margin": margin_bad / max(local_n, 1),
        "violation_long": long_bad / max(long_n, 1),
        "roughness": rough_num / max(rough_den, 1e-12),
    }


@torch.no_grad()
def evaluate_progress(
    model: ProgressCVAE,
    rows: dict[str, torch.Tensor],
    cfg: ProgressConfig,
    scale_px: float = 256.0,
    full: bool = False,
) -> dict:
    """Score the encoder against the held-out phase.

    This is the **only** function that touches ``rows["phase_holdout"]``.
    """
    model.eval()
    enc = encode_rows(model, rows)
    z, truth = enc["z_phase"], rows["phase_holdout"]
    groups = _episode_groups(rows)

    per_ep = [spearman(z[lo:hi], truth[lo:hi]) for _, lo, hi in groups]
    finite = [r for r in per_ep if not math.isnan(r)]
    pooled = spearman(z, truth)
    gen = torch.Generator(device=str(z.device)).manual_seed(cfg.seed)
    pos_hat, v = model.decode(z, enc["mu"], rows["cond"])
    out: dict = {
        "n_rows": int(z.numel()),
        "n_episodes": len(groups),
        "spearman_per_episode_mean": sum(finite) / len(finite) if finite else float("nan"),
        "spearman_pooled": pooled,
        "recon": float(model.recon_error(rows["obs"], pos_hat, v).mean()),
        **_ablations(model, rows, enc, gen),
        **_violations(rows, z, cfg.kappa, seed=cfg.seed),
    }
    out["calibration_gap"] = out["spearman_per_episode_mean"] - pooled
    if not full:
        return out

    enc_nc = encode_rows(model, rows, cond=False)
    per_ep_nc = [spearman(enc_nc["z_phase"][lo:hi], truth[lo:hi]) for _, lo, hi in groups]
    finite_nc = [r for r in per_ep_nc if not math.isnan(r)]

    terciles = {}
    for name, (a, b) in {"t1": (0.0, 1 / 3), "t2": (1 / 3, 2 / 3), "t3": (2 / 3, 1.01)}.items():
        keep = (truth >= a) & (truth < b)
        rho = [spearman(z[lo:hi][keep[lo:hi]], truth[lo:hi][keep[lo:hi]]) for _, lo, hi in groups]
        rho = [r for r in rho if not math.isnan(r)]
        terciles[f"spearman_{name}"] = sum(rho) / len(rho) if rho else float("nan")

    truth_emb = embed_state(rows["obs"], model.pos)
    unit = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    d_theta = torch.atan2(unit[..., 1], unit[..., 0]) - torch.atan2(
        truth_emb[..., 5], truth_emb[..., 4]
    )
    d_theta = torch.atan2(torch.sin(d_theta), torch.cos(d_theta))
    kl_dim = 0.5 * (enc["mu"].pow(2) + enc["logvar"].exp() - 1.0 - enc["logvar"]).mean(0)
    spans = [float(z[lo:hi].max() - z[lo:hi].min()) for _, lo, hi in groups if hi > lo]

    out.update({
        "spearman_per_episode_median": _percentile(per_ep, 0.5),
        "spearman_per_episode_p10": _percentile(per_ep, 0.10),
        "spearman_per_episode_ci95": _bootstrap_ci(per_ep, seed=cfg.seed),
        "spearman_with_cond": out["spearman_per_episode_mean"],
        "spearman_no_cond": sum(finite_nc) / len(finite_nc) if finite_nc else float("nan"),
        **terciles,
        "ood_p50": float(enc["ood"].quantile(0.50)),
        "ood_p95": float(enc["ood"].quantile(0.95)),
        "ood_max": float(enc["ood"].max()),
        "kl_per_dim": sorted(kl_dim.tolist(), reverse=True),
        "kl_total": float(kl_dim.sum()),
        "active_units": int((enc["mu"].var(0) > 0.01).sum()),
        "sat_frac": float((enc["u"].abs() > model.sat_limit).to(z.dtype).mean()),
        "z_phase_span_mean": sum(spans) / len(spans) if spans else float("nan"),
        "z_phase_span_min": min(spans) if spans else float("nan"),
        "rmse_agent_px": scale_px * float(
            (pos_hat[..., :2] - truth_emb[..., :2]).pow(2).sum(-1).mean().sqrt()
        ),
        "rmse_block_px": scale_px * float(
            (pos_hat[..., 2:4] - truth_emb[..., 2:4]).pow(2).sum(-1).mean().sqrt()
        ),
        "rmse_theta_deg": math.degrees(float(d_theta.pow(2).mean().sqrt())),
    })
    return out


# -- null baselines ----------------------------------------------------------------------


def phase_baselines(
    ep: PushTEpisodes,
    prep: PushTPrepared,
    eval_ids: torch.Tensor,
    obs_steps: int = 2,
    device: str | torch.device = "cpu",
) -> dict:
    """Two label-free nulls the CVAE has to beat before any of this is a result.

    ``rho_geom`` is the honest mechanism stated as a baseline: the goal pose is a *constant*
    across every Push-T episode, and competent demonstrators monotonically reduce the SE(2)
    distance from the block to it, so a hand-coded ``-d_SE2(block, goal)`` is already a progress
    measure. ``rho_linear`` is the linear-probe ceiling on the same window the CVAE sees, fitted
    on training rows only.

    A CVAE that does not clearly beat ``max(rho_geom, rho_linear)`` has not shown anything, and
    this costs no training at all -- run it first.
    """
    train_rows = progress_rows(ep, to=obs_steps, device=device, episode_ids=prep.train_ids)
    rows = progress_rows(ep, to=obs_steps, device=device, episode_ids=eval_ids)
    pos = prep.normalizer("agent_pos")

    goal = torch.tensor(GOAL_POSE, dtype=torch.float32, device=device)
    block = rows["obs"][:, -1, 2:5]  # (R, 3) current block pose
    d_goal = -twist_norm(se2_log(block, goal.expand_as(block)), rot_scale=ROT_SCALE_PX_PER_RAD)

    feat_train = embed_state(train_rows["obs"], pos).flatten(1)  # (N, To*6)
    feat_eval = embed_state(rows["obs"], pos).flatten(1)
    ones_train = torch.ones((feat_train.shape[0], 1), device=device)
    ones_eval = torch.ones((feat_eval.shape[0], 1), device=device)
    a = torch.cat([feat_train, ones_train], dim=-1).to(torch.float64)
    w = torch.linalg.lstsq(a, train_rows["phase_holdout"][:, None].to(torch.float64)).solution
    linear = (torch.cat([feat_eval, ones_eval], dim=-1).to(torch.float64) @ w).squeeze(-1)

    groups = _episode_groups(rows)
    truth = rows["phase_holdout"]

    def per_episode(values):
        rho = [spearman(values[lo:hi], truth[lo:hi]) for _, lo, hi in groups]
        rho = [r for r in rho if not math.isnan(r)]
        return sum(rho) / len(rho) if rho else float("nan")

    return {
        "rho_geom": per_episode(d_goal),
        "rho_geom_pooled": spearman(d_goal, truth),
        "rho_linear": per_episode(linear),
        "rho_linear_pooled": spearman(linear, truth),
    }


# -- training ----------------------------------------------------------------------------


def train_progress(
    ep: PushTEpisodes,
    prep: PushTPrepared,
    cfg: ProgressConfig,
    out_dir: str | Path | None = None,
    eval_ids: torch.Tensor | None = None,
    verbose: bool = True,
) -> tuple[ProgressCVAE, dict]:
    """Train one progress CVAE; returns the (EMA) model and its full evaluation.

    With ``out_dir``, writes ``config.json``, ``log.csv``, ``final.pt`` and -- last, as the
    finished-run marker -- ``metrics.json``, the same contract as
    :func:`fm_bc.skate.baseline_training.train_baseline`.
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    if eval_ids is None:
        eval_ids = prep.held_out_ids

    train = progress_rows(ep, to=cfg.obs_steps, device=device, episode_ids=prep.train_ids)
    held = progress_rows(ep, to=cfg.obs_steps, device=device, episode_ids=eval_ids)
    pos = prep.normalizer("agent_pos", cfg.normalizer)
    scale_px = float(prep.normalizer("state", cfg.normalizer).scale[:2].mean())

    model = ProgressCVAE(
        pos_norm=pos, obs_steps=cfg.obs_steps, z_style=cfg.z_style, hidden=cfg.hidden,
        activation=cfg.activation, beta=cfg.beta, free_bits=cfg.free_bits,
        mono_weight=cfg.mono_weight, kappa=cfg.kappa, kappa_hi=cfg.kappa_hi,
        sat_weight=cfg.sat_weight, style_dropout=cfg.style_dropout,
    ).to(device)
    ema = copy.deepcopy(model).requires_grad_(False) if cfg.ema else model
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(cfg, s))

    n_rows = train["obs"].shape[0]
    longest = int(ep.lengths.max())
    info = {
        "train_config": asdict(cfg),
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_train_rows": n_rows,
        "n_eval_rows": int(held["obs"].shape[0]),
        "train_episodes": prep.train_ids.tolist(),
        "eval_episodes": eval_ids.tolist(),
        "val_episodes": prep.val_ids.tolist(),
        # The smallest per-step rise the longest episode is being asked for.
        "min_step_margin": cfg.kappa / max(longest - 1, 1),
        "w_theta": float(model.w_theta),
        "data_meta": ep.meta,
    }
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(info, indent=2, default=str))

    log: list[dict] = []
    t0 = time.time()
    model.train()
    for step in range(1, cfg.steps + 1):
        idx = torch.randint(n_rows, (cfg.batch_size,), device=device, generator=gen)
        partner, valid, margin, _ = sample_pairs(
            train, idx, gap=cfg.gap, stride=cfg.stride, kappa=cfg.kappa, generator=gen
        )
        keep = None
        if cfg.cond_dropout > 0.0:
            keep = torch.rand(idx.shape, generator=gen, device=device) >= cfg.cond_dropout
        # Only obs and cond reach the model; phase_holdout is never read in this loop.
        loss, extras = model.loss(
            train["obs"][idx], train["cond"][idx],
            obs_next=train["obs"][partner], valid=valid, margin=margin,
            cond_keep=keep, beta_scale=kl_scale(cfg, step), generator=gen,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        if cfg.ema:
            _ema_update(ema, model, ema_decay(cfg, step))

        if step % cfg.log_every == 0 or step == 1:
            row = {"step": step, "seconds": round(time.time() - t0, 2),
                   "lr": sched.get_last_lr()[0], "total_loss": float(loss.detach()), **extras}
            if step % cfg.eval_every == 0 or step == cfg.steps:
                ev = evaluate_progress(ema, held, cfg, scale_px, full=False)
                row.update({
                    "val_spearman": ev["spearman_per_episode_mean"],
                    "val_violation_local": ev["violation_local"],
                    "val_phase_ablation": ev["phase_ablation"],
                    "val_recon": ev["recon"],
                })
                if verbose:
                    print(f"  step {step:>6}  loss {row['total_loss']:.4f}  "
                          f"rho {ev['spearman_per_episode_mean']:.3f}  "
                          f"viol {ev['violation_local']:.3f}  "
                          f"abl {ev['phase_ablation']:.1f}%  "
                          f"AU {row['active_units']}  ({row['seconds']:.0f}s)")
                model.train()
            log.append(row)

    final = evaluate_progress(ema, held, cfg, scale_px, full=True)
    final.update(phase_baselines(ep, prep, eval_ids, cfg.obs_steps, device))
    final["val_split"] = evaluate_progress(
        ema, progress_rows(ep, to=cfg.obs_steps, device=device, episode_ids=prep.val_ids),
        cfg, scale_px, full=False,
    )
    final.update(info)
    final["train_seconds"] = time.time() - t0
    if out is not None:
        _write_log(out / "log.csv", log)
        ema.save(out / "final.pt", info=info)
        # Last: metrics.json marks a finished run.
        (out / "metrics.json").write_text(json.dumps(final, indent=2, default=str))
    return ema, final
