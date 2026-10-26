"""Train an action-chunk baseline -- Diffusion Policy, MIP, MeanFlow or ACT -- on the same
demonstrations as the FM / MLP policies, and evaluate it with the same rollouts and metrics.

Each method keeps its own paper's optimization recipe (``METHOD_DEFAULTS``), and any field left
``None`` on :class:`BaselineConfig` is filled in from it:

* ``dp`` / ``meanflow`` -- Diffusion Policy's low-dim UNet recipe: AdamW 1e-4, betas
  (0.95, 0.999), weight decay 1e-6, batch 256, 500 warmup steps then cosine decay, and the
  power-schedule EMA (``1 - (1 + step)^-0.75``, capped at 0.9999).
* ``mip`` -- the MIP repo's ``optimization/default.yaml``: AdamW 1e-4 at a constant rate,
  weight decay 1e-5, batch 1024, gradient clipping at 10, EMA 0.995.
* ``act`` -- LeRobot's ACT config: AdamW 1e-5 at a constant rate, weight decay 1e-4, batch 8,
  no EMA.

Only the step budget is shared, so every baseline sees the same number of *updates* -- but not
the same number of *samples*, because the batch sizes differ by up to 128x. That is the usual
consequence of letting each method keep its recipe, and it matters most for ACT, whose batch of 8
is 32x smaller than Diffusion Policy's: ``--batch-size 256`` gives the data-matched control.

Evaluation always uses the EMA weights and ``final.pt`` is the EMA model; under ``ema="none"``
the EMA is an exact copy of the trained model, so this stays true without a special case.
"""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from .baselines import ChunkPolicy, make_policy
from .chunks import make_windows
from .dataset import TrajectoryDataset
from .evaluation import evaluate_policy
from .training import BASELINE_METHODS, _write_log

__all__ = ["METHOD_DEFAULTS", "BaselineConfig", "baseline_run_dir", "ema_decay", "lr_factor",
           "train_baseline"]

_DP_RECIPE = {"batch_size": 256, "lr": 1e-4, "weight_decay": 1e-6, "betas": (0.95, 0.999),
              "warmup_steps": 500, "lr_schedule": "cosine", "ema": "power", "grad_clip": None}
METHOD_DEFAULTS = {
    "dp": _DP_RECIPE,
    "meanflow": _DP_RECIPE,
    "mip": {"batch_size": 1024, "lr": 1e-4, "weight_decay": 1e-5, "betas": (0.9, 0.999),
            "warmup_steps": 0, "lr_schedule": "constant", "ema": "fixed", "grad_clip": 10.0},
    # LeRobot's ACT defaults: AdamW 1e-5 at a constant rate, weight decay 1e-4, batch 8.
    # The batch size is ACT's own and is 32x smaller than Diffusion Policy's, so at a shared step
    # budget ACT sees 32x fewer samples -- pass --batch-size 256 for the data-matched control.
    #
    # `ema` is the single departure from that config, which trains without one. Nav has no
    # best-checkpoint selection -- `final.pt` is whatever step 100000 lands on -- and DP and MIP
    # are both scored on EMA weights, so a raw snapshot would grade ACT under a different
    # protocol than the arms beside it in the figure. MIP's fixed 0.995 (the `ema_rate` default)
    # is the smallest fix. At batch 8 it averages over ~1600 samples, so it is very nearly a
    # no-op; the reported nav runs are the published recipe in every other respect.
    #
    # Two alternatives were measured over 10 seeds x 3 scenarios and both made the endpoint
    # WORSE, so neither is the default:
    #   * batch 256 (DP's sample budget, 25.6M vs 800k): dual_goal 0.950 -> 0.410.
    #   * batch 256 + cosine decay: single_path 0.315 -> 0.175, dual_path 0.494 -> 0.416.
    # Cosine did converge -- the second-half swing fell from 0.496 to 0.317 on dual_path -- but
    # ACT degrades with training here: on dual_path it peaks at 0.992 success by step 40000 and
    # falls to 0.504 by 100000 while its training loss stays flat, and dual_goal's MMD rises
    # 0.050 -> 0.372 over the same span. The excursions to ~0.97 are a transient it passes
    # through, not a level it holds. See runs_actb256/ if those runs are still around.
    "act": {"batch_size": 8, "lr": 1e-5, "weight_decay": 1e-4, "betas": (0.9, 0.999),
            "warmup_steps": 0, "lr_schedule": "constant", "ema": "fixed", "grad_clip": None},
}


@dataclass
class BaselineConfig:
    method: str = "dp"
    steps: int = 100000
    # None: the method's default from METHOD_DEFAULTS (filled in by __post_init__).
    lr: float | None = None
    batch_size: int | None = None
    weight_decay: float | None = None
    betas: tuple | None = None
    warmup_steps: int | None = None
    lr_schedule: str | None = None  # "cosine" (after warmup) or "constant"
    ema: str | None = None  # "power" (Diffusion Policy's EMAModel) or "fixed" (ema_rate)
    ema_rate: float = 0.995
    ema_power: float = 0.75
    ema_max: float = 0.9999
    grad_clip: float | None = None
    obs_steps: int = 2
    horizon: int = 16
    action_steps: int = 8
    model: dict = field(default_factory=dict)  # extra network kwargs, e.g. smaller test nets
    n_train: int | None = None  # None: every demonstration
    k: int = 16  # kNN rank of the rollouts' divergence (and of prepare_training_set)
    eval_every: int = 5000
    log_every: int = 100
    checkpoint_every: int = 1000  # overwrite last.pt this often; 0 disables
    mmd_bandwidth: float = 0.3
    clamp_controls: bool = True
    seed: int = 0
    device: str = "cuda"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.method not in BASELINE_METHODS:
            raise ValueError(f"method must be one of {BASELINE_METHODS}, got {self.method!r}")
        for key, value in METHOD_DEFAULTS[self.method].items():
            if getattr(self, key) is None:
                setattr(self, key, value)
        self.betas = tuple(self.betas)
        if self.lr_schedule not in ("cosine", "constant"):
            raise ValueError(f"lr_schedule must be cosine or constant, got {self.lr_schedule!r}")
        if self.ema not in ("power", "fixed", "none"):
            raise ValueError(f"ema must be power, fixed or none, got {self.ema!r}")


def baseline_run_dir(root: str | Path, scenario: str, n_train: int, cfg: BaselineConfig) -> Path:
    return Path(root) / scenario / f"n{n_train}" / cfg.method / f"seed{cfg.seed}"


def lr_factor(cfg: BaselineConfig, step: int) -> float:
    """Multiplier of ``cfg.lr`` after ``step`` optimizer steps: diffusers'
    ``get_cosine_schedule_with_warmup`` (linear warmup, then a half cosine to 0), or 1."""
    if step < cfg.warmup_steps:
        return step / max(1, cfg.warmup_steps)
    if cfg.lr_schedule == "constant":
        return 1.0
    progress = (step - cfg.warmup_steps) / max(1, cfg.steps - cfg.warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def ema_decay(cfg: BaselineConfig, step: int) -> float:
    """EMA decay at optimizer step ``step`` (1-based). ``"power"`` is Diffusion Policy's
    ``EMAModel.get_decay`` with ``update_after_step=0, inv_gamma=1``: 0 at the first step (the
    EMA starts as a copy), then ``1 - (1 + s)^-power`` capped at ``ema_max``.

    ``"none"`` is a decay of exactly 0 at every step, which makes ``_ema_update`` an exact copy --
    so ``final.pt`` is the trained model and the rest of the loop is untouched. ACT is trained
    without an EMA, and this says so rather than spelling it ``ema="fixed", ema_rate=0.0``.
    """
    if cfg.ema == "none":
        return 0.0
    if cfg.ema == "fixed":
        return cfg.ema_rate
    s = step - 1
    if s <= 0:
        return 0.0
    return min(max(1 - (1 + s) ** -cfg.ema_power, 0.0), cfg.ema_max)


@torch.no_grad()
def _ema_update(ema: ChunkPolicy, model: ChunkPolicy, decay: float) -> None:
    for p_ema, p in zip(ema.parameters(), model.parameters()):
        p_ema.mul_(decay).add_(p.detach(), alpha=1 - decay)


def train_baseline(
    ds_train: TrajectoryDataset,
    ds_test: TrajectoryDataset,
    cfg: BaselineConfig,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[ChunkPolicy, dict]:
    """Train one baseline; returns its EMA model and final (full) evaluation.

    With ``out_dir``, writes the same files as :func:`fm_bc.skate.training.train`: ``config.json``,
    ``log.csv`` (the objective as ``total_loss``), ``final.pt``, ``rollouts.npz`` and, last,
    ``metrics.json``.
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    limits = ds_train.limits
    model = make_policy(cfg.method, obs_steps=cfg.obs_steps, horizon=cfg.horizon,
                        action_steps=cfg.action_steps,
                        limits={"v_max": limits.v_max, "v_min": limits.v_min,
                                "omega_max": limits.omega_max},
                        **cfg.model).to(device)
    ema = copy.deepcopy(model).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas,
                            weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(cfg, s))
    windows = make_windows(ds_train, cfg.obs_steps, cfg.horizon, device)
    n_rows = windows["obs"].shape[0]

    out = Path(out_dir) if out_dir is not None else None
    info = {
        "train_config": asdict(cfg),
        "scenario": ds_train.scenario,
        "n_train": len(ds_train),
        "n_test": len(ds_test),
        "n_params": sum(p.numel() for p in model.parameters()),
        "div_config": ds_train.div_config,
    }
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(info, indent=2))

    def evaluate(full: bool) -> tuple[dict, dict]:
        return evaluate_policy(ema, ds_test, bandwidth=cfg.mmd_bandwidth, k=cfg.k, full=full,
                               clamp=cfg.clamp_controls, device=device, sample_seed=cfg.seed)

    log: list[dict] = []
    start_step = 0
    # Resume a run killed mid-flight. metrics.json is what marks a run finished (and what
    # train_baselines.py skips on), so an interrupted run is re-entered here and picks up from
    # its last checkpoint instead of restarting from zero. Everything the loop advances is
    # restored -- weights, EMA, optimizer moments, LR schedule and the batch-sampling generator
    # -- so resuming is equivalent to never having stopped, up to the steps since the last write.
    ckpt_path = (out / "last.pt") if out is not None else None
    if ckpt_path is not None and ckpt_path.exists():
        state = torch.load(ckpt_path, map_location=device)
        resume = state.get("resume")
        if resume is not None:
            model.load_state_dict(resume["model"])
            ema.load_state_dict(state["state_dict"])
            opt.load_state_dict(resume["opt"])
            sched.load_state_dict(resume["sched"])
            gen.set_state(resume["gen"].cpu())
            # The explicit generator only drives batch sampling. Dropout and ACT's CVAE
            # reparameterization draw from the global streams, so those have to come back too
            # or a resumed run silently diverges from the uninterrupted one.
            torch.set_rng_state(resume["cpu_rng"].cpu())
            if resume.get("cuda_rng") is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state(resume["cuda_rng"].cpu())
            start_step, log = resume["step"], list(resume["log"])
            if verbose:
                print(f"    resuming from step {start_step} of {cfg.steps}", flush=True)

    def _checkpoint(step: int) -> None:
        """Overwrite last.pt: the EMA policy (so load_policy reads it like any checkpoint)
        plus everything needed to restart the loop exactly where it stopped."""
        ema.save(ckpt_path, info=info, resume={
            "step": step, "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "gen": gen.get_state(), "log": log,
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None})

    # Back-date the clock by the time already logged, so `seconds` and train_seconds stay
    # cumulative across a resume rather than restarting at the interruption.
    t0 = time.time() - (log[-1]["seconds"] if log else 0.0)
    model.train()
    for step in range(start_step + 1, cfg.steps + 1):
        idx = torch.randint(n_rows, (cfg.batch_size,), device=device, generator=gen)
        loss, extras = model.loss(windows["obs"][idx], windows["act"][idx])
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()
        _ema_update(ema, model, ema_decay(cfg, step))

        if step % cfg.log_every == 0 or step == 1:
            row = {"step": step, "seconds": round(time.time() - t0, 2),
                   "lr": sched.get_last_lr()[0], "cdm_weight": 0.0, "total_loss": loss.item()}
            if step % cfg.eval_every == 0 or step == cfg.steps:
                ev, _ = evaluate(full=False)
                row.update({key: ev[key] for key in
                            ("mmd_mean", "mmd_final", "collision_rate", "success_rate", "mode_tv",
                             "saturation_rate")})
                if verbose:
                    print(f"  step {step:>6}  loss {row['total_loss']:.4f}  "
                          f"mmd {ev['mmd_mean']:.4f}  coll {ev['collision_rate']:.3f}  "
                          f"succ {ev['success_rate']:.3f}  "
                          f"modes {[round(f, 2) for f in ev['mode_fractions']]}  "
                          f"({row['seconds']:.0f}s)")
            log.append(row)
            del extras

        if ckpt_path is not None and cfg.checkpoint_every and step % cfg.checkpoint_every == 0:
            _write_log(out / "log.csv", log)  # so a killed run keeps its curve, not just weights
            _checkpoint(step)

    final, arrays = evaluate(full=True)
    final.update(info)
    final["train_seconds"] = time.time() - t0
    if out is not None:
        _write_log(out / "log.csv", log)
        ema.save(out / "final.pt", info=info)
        np.savez_compressed(out / "rollouts.npz", **arrays)
        # Last: metrics.json marks a finished run (train_baselines skips on it).
        (out / "metrics.json").write_text(json.dumps(final, indent=2))
    return ema, final
