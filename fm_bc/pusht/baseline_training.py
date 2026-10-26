"""Training of the Push-T chunk baselines (DP, MIP, ACT) on the MLP arms' pipeline.

Same recipe as :mod:`fm_bc.pusht.policy_training` -- Diffusion Policy's ``pusht_lowdim``
optimizer (AdamW 1e-4, betas (0.95, 0.999), weight decay 1e-6, cosine schedule with 500 warmup
steps stepped per batch), batch 256, EMA (power 0.75, max 0.9999), the same number of epochs, a
checkpoint every 50 epochs, and ``best.pt`` = the EMA weights at the lowest EMA validation loss --
so the only differences from the MLP arms are the model and what it predicts: a chunk of 16
absolute pixel targets from the last 2 states, 8 of them executed per re-plan.

The validation loss is each model's own training objective on the validation windows, evaluated
with a **fixed generator** (the same noise and timesteps every epoch), so the selection criterion
is comparable across epochs. For ACT the generator drives the CVAE reparameterization draw, which
is why :class:`fm_bc.pusht.baselines.PushTACTPolicy` threads it all the way into the latent.

``use_ema=False`` (ACT is trained without an EMA) makes ``ema`` the model itself, so ``val_loss``
and ``val_loss_ema`` coincide and ``best.pt`` is the raw model at its lowest validation loss.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from .baselines import PUSHT_BASELINES, PushTChunkPolicy, make_pusht_baseline
from .chunks import make_windows
from .data import PushTEpisodes
from .prepared import PushTPrepared
from .progress_training import _ema_update, ema_decay
from .policy_training import lr_factor

__all__ = ["BaselineConfig", "baseline_run_dir", "train_pusht_baseline", "validation_loss"]

_LOG_FIELDS = ["epoch", "global_step", "seconds", "lr", "train_loss", "val_loss", "val_loss_ema",
               "is_best"]


@dataclass
class BaselineConfig:
    kind: str = "dp"                      # dp | mip | act
    obs_steps: int = 2
    horizon: int = 16
    action_steps: int = 8
    normalizer: str = "minmax"            # of the observation; actions use the action box
    num_epochs: int = 500
    batch_size: int = 256
    lr: float = 1e-4
    betas: tuple[float, float] = (0.95, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-6
    lr_warmup_steps: int = 500
    use_ema: bool = True
    ema_power: float = 0.75
    ema_max: float = 0.9999
    val_every: int = 1
    checkpoint_every: int = 50
    seed: int = 0
    device: str = "cuda"
    model_kwargs: dict = field(default_factory=dict)


def baseline_run_dir(root: str | Path, cfg: BaselineConfig) -> Path:
    return Path(root) / "pusht" / "policy" / cfg.kind / f"seed{cfg.seed}"


def _windows(ep: PushTEpisodes, ids, cfg: BaselineConfig) -> dict[str, torch.Tensor]:
    w = make_windows(ep, to=cfg.obs_steps, tp=cfg.horizon, device=cfg.device, episode_ids=ids)
    return {"obs": w["obs"].float(), "act": w["act"].float()}


@torch.no_grad()
def validation_loss(model: PushTChunkPolicy, data: dict, seed: int = 0,
                    batch_size: int = 512) -> float:
    """The model's own objective on ``data``, with a fixed generator (see the module docstring)."""
    was_training = model.training
    model.eval()
    device = data["obs"].device
    total, n = 0.0, data["obs"].shape[0]
    for lo in range(0, n, batch_size):
        gen = torch.Generator(device=device).manual_seed(seed + lo)
        obs, act = data["obs"][lo:lo + batch_size], data["act"][lo:lo + batch_size]
        loss, _ = model.loss_normalized(obs, model.act.apply(act), generator=gen)
        total += float(loss) * obs.shape[0]
    model.train(was_training)
    return total / n


def train_pusht_baseline(
    ep: PushTEpisodes,
    prep: PushTPrepared,
    cfg: BaselineConfig,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[PushTChunkPolicy, dict]:
    """Train one baseline; returns the best (lowest EMA validation loss) model and its metrics."""
    if cfg.kind not in PUSHT_BASELINES:
        raise ValueError(f"kind must be one of {tuple(PUSHT_BASELINES)}, got {cfg.kind!r}")
    torch.manual_seed(cfg.seed)
    device = cfg.device
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    train = _windows(ep, prep.train_ids, cfg)
    val = _windows(ep, prep.val_ids, cfg)
    norms = {"pos_norm": prep.normalizer("agent_pos", cfg.normalizer),
             "act_norm": prep.normalizer("action", "box")}
    model = make_pusht_baseline(cfg.kind, obs_steps=cfg.obs_steps, horizon=cfg.horizon,
                                action_steps=cfg.action_steps, **norms,
                                **cfg.model_kwargs).to(device)
    ema = (make_pusht_baseline(cfg.kind, **model.config()).to(device).requires_grad_(False)
           if cfg.use_ema else model)
    if cfg.use_ema:
        ema.load_state_dict(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
                            weight_decay=cfg.weight_decay)

    n_rows = train["obs"].shape[0]
    steps_per_epoch = math.ceil(n_rows / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.num_epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(cfg, s, total_steps))

    info = {
        "method": cfg.kind,
        "train_config": asdict(cfg),
        "pipeline": "Diffusion Policy pusht_lowdim workspace (fm_bc.pusht.baseline_training)",
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_train_rows": n_rows,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "train_episodes": prep.train_ids.tolist(),
        "val_episodes": prep.val_ids.tolist(),
        "inputs": f"last {cfg.obs_steps} states [agent xy_n, block xy_n, cos, sin]",
        "target": f"chunk of {cfg.horizon} absolute pixel targets, {cfg.action_steps} executed",
        "selection": ("best.pt = EMA model at the lowest EMA validation loss (its own objective)"
                      if cfg.use_ema else
                      "best.pt = model at the lowest validation loss (its own objective; no EMA)"),
    }
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(info, indent=2))

    log: list[dict] = []
    best = {"val_loss_ema": math.inf, "epoch": -1}
    t0, global_step = time.time(), 0
    model.train()
    for epoch in range(cfg.num_epochs):
        perm = torch.randperm(n_rows, device=device, generator=gen)
        total = 0.0
        for lo in range(0, n_rows, cfg.batch_size):
            idx = perm[lo:lo + cfg.batch_size]
            loss, _ = model.loss(train["obs"][idx], train["act"][idx])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {global_step}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            global_step += 1
            if cfg.use_ema:
                _ema_update(ema, model, ema_decay(cfg, global_step))
            total += float(loss)
        row = {"epoch": epoch, "global_step": global_step, "seconds": round(time.time() - t0, 2),
               "lr": sched.get_last_lr()[0], "train_loss": total / steps_per_epoch, "is_best": 0}
        if epoch % cfg.val_every == 0 or epoch == cfg.num_epochs - 1:
            row["val_loss"] = validation_loss(model, val, seed=cfg.seed)
            row["val_loss_ema"] = validation_loss(ema, val, seed=cfg.seed)
            if row["val_loss_ema"] < best["val_loss_ema"]:
                best = {"val_loss_ema": row["val_loss_ema"], "epoch": epoch}
                row["is_best"] = 1
                if out is not None:
                    ema.save(out / "best.pt", info=info, epoch=epoch,
                             val_loss=row["val_loss_ema"])
        if out is not None and (epoch + 1) % cfg.checkpoint_every == 0:
            ema.save(out / f"epoch_{epoch + 1:04d}.pt", info=info, epoch=epoch)
        log.append(row)
        if verbose and (epoch % 25 == 0 or epoch == cfg.num_epochs - 1):
            print(f"  epoch {epoch:>4}  step {global_step:>6}  train {row['train_loss']:.4f}  "
                  f"val {row.get('val_loss', math.nan):.4f} "
                  f"(ema {row.get('val_loss_ema', math.nan):.4f})  best epoch {best['epoch']}  "
                  f"({row['seconds']:.0f}s)", flush=True)

    metrics = {**info, "best": best, "last": {"val_loss_ema": log[-1].get("val_loss_ema"),
                                              "epoch": cfg.num_epochs - 1},
               "train_seconds": time.time() - t0}
    if out is not None:
        ema.save(out / "last.pt", info=info, epoch=cfg.num_epochs - 1)
        with open(out / "log.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_LOG_FIELDS, extrasaction="ignore")
            w.writeheader()
            w.writerows(log)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return ema, metrics
