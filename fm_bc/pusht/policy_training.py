"""Train and score the Push-T policy arms: the plain arm, +phase and +phase+score.

``cfg.arch`` picks the backbone (:data:`fm_bc.pusht.policy.ARCHS`) -- ``"mlp"``
(:class:`~fm_bc.pusht.policy.PushTMLP`, the default and everything trained so far) or
``"transformer"`` (:class:`~fm_bc.pusht.policy.PushTTransformer`). The loop below is
backbone-agnostic: it only ever calls ``model(obs, phase)``, and the score regularizer acts on
``u``, so every arm exists for both. A non-MLP backbone appends :data:`ARCH_SUFFIX` to the run
name, leaving existing run directories untouched.

**The training pipeline follows Diffusion Policy's Push-T low-dim workspace** (Chi et al. 2023,
``train_diffusion_unet_lowdim_workspace.yaml`` + ``task/pusht_lowdim.yaml``), not the skate
study's step-budget recipe:

* **Splits** -- whatever the prepared artifact holds. DP's own (``seed 42``, ``val_ratio 0.02``,
  ``max_train_episodes 90``) gives 90 train / 4 val, and its test is 50 environment rollouts, but
  4 validation episodes proved too few to select a checkpoint. The paper uses the 90/10 ratio
  split (``pusht_prepared_9010.npz``, 185 / 21): the validation episodes select the checkpoint
  and the closed-loop rollouts in :mod:`fm_bc.pusht.sim` are the test. The earlier 70/20/10 split
  (``pusht_prepared_702010.npz``, 144 / 41 / 21) also holds out test episodes
  (:attr:`PushTPrepared.held_out_ids`), which never enter training or model selection.
* **Epochs** -- every epoch is one shuffled pass over all training rows in batches of 256.
* **Optimizer** -- AdamW, lr 1e-4, betas (0.95, 0.999), eps 1e-8, weight decay 1e-6; cosine
  schedule with 500 warmup steps, stepped every batch (diffusers' ``get_scheduler("cosine")``).
* **EMA** -- DP's ``EMAModel`` (power 0.75, max 0.9999), updated every batch.
* **Validation** -- the validation loss every ``val_every`` epochs (DP: 1). DP computes it with
  the raw model; both the raw and the EMA model's losses are logged.
* **Checkpoints** -- DP keeps the top-k by rollout success, which needs a simulator. Here
  ``best.pt`` is the EMA model at the lowest EMA validation loss, and the EMA model is also saved
  every ``checkpoint_every`` epochs (DP: 50) and at the end (``last.pt``, DP's
  ``save_last_ckpt``).
* **Normalizer** -- DP's ``limits`` mode is min-max to [-1, 1]; that is ``minmax`` here. DP fits it
  on the whole replay buffer, validation episodes included; here it is fitted on training
  episodes only.

The network regresses the agent's phase velocity ``u = dx/drho`` (see :mod:`fm_bc.pusht.policy`)
and the loss is

    L = E |u_hat - u|^2  +  lambda(step) * E |(u - u_hat) . grad log p_rho(x_agent)|

in normalized position units, with ``lambda`` zero for the MLP and MLP+phase arms and, for
MLP+phase+score and MLP+score, a cosine from ``score_weight`` to 0 over the first ``score_anneal_frac`` of the
optimizer steps (the skate study's ``score_w0.05_cos0.5``); ``score_anneal_frac = 0`` holds
``lambda`` at ``score_weight`` for the whole run instead (``score_w<λ>_const``), the ablation of
the schedule itself. The validation loss is the velocity
MSE alone, so every arm is selected on the same quantity. Run names match the skate ones
(``mlp``, ``fm``, ``score_w<λ>_cos<frac>``).

**Phase** is the demonstration's own normalized time ``t / (L - 1)`` -- an oracle, exact on
demonstrations and unknown in a closed-loop rollout.

**MLP+score** is the no-phase ablation: the network never sees ``rho``, but the regularizer is
unchanged -- the score is still that of the demonstrations at the row's phase, a training-time
signal only, exactly as the skate study's ``mlp_*`` arms use the stored per-phase score.

**Score** is precomputed once per training row: the agent-position kernel score at the row's
phase, against the training episodes' agent positions on the shared phase grid at the nearest
grid phase.

Theseus-free, like the rest of :mod:`fm_bc.pusht`; the EMA schedule is shared with
:mod:`fm_bc.pusht.progress_training`, which restates DP's.
"""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Union

import torch

from .chunks import make_windows
from .data import PushTEpisodes
from .policy import ARCHS, PushTMLP, PushTTransformer, agent_score, make_pusht_policy
from .prepared import PushTPrepared
from .progress_training import _ema_update, ema_decay

__all__ = ["PolicyConfig", "METHODS", "ARCH_SUFFIX", "method_name", "policy_run_dir",
           "policy_rows", "score_weight_at", "lr_factor", "evaluate_policy_offline",
           "validation_loss", "train_pusht_policy"]

# Either backbone: the loop only ever calls model(obs, phase), so it is backbone-agnostic.
# typing.Union, not `A | B`: this is an eager assignment and the project supports Python 3.8.
Policy = Union[PushTMLP, PushTTransformer]


@dataclass
class PolicyConfig:
    use_phase: bool = True
    score_weight: float = 0.0  # > 0: MLP+phase+score, or MLP+score without use_phase
    score_anneal_frac: float = 0.5  # cosine to 0 over this fraction of steps; 0: constant
    bandwidth_px: float = 32.0  # agent-score kernel; between NN spacing (16) and Silverman (45)
    obs_steps: int = 1  # inputs [o, x, rho] at the current step only
    normalizer: str = "minmax"  # DP's "limits"
    arch: str = "mlp"  # backbone: fm_bc.pusht.policy.ARCHS
    hidden: tuple = (512, 512, 512, 512)  # arch="mlp" only
    activation: str = "silu"  # arch="mlp" only
    # Backbone kwargs for a non-MLP arch (hidden_dim, num_layers, num_heads, dropout, ...).
    net_kwargs: dict = field(default_factory=dict)
    # -- DP's optimization
    num_epochs: int = 500
    batch_size: int = 256
    lr: float = 1e-4
    betas: tuple = (0.95, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-6
    lr_warmup_steps: int = 500
    use_ema: bool = True
    ema_power: float = 0.75
    ema_max: float = 0.9999
    # -- DP's loop control, in epochs
    val_every: int = 1
    checkpoint_every: int = 50
    seed: int = 0
    device: str = "cuda"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.score_anneal_frac <= 1:
            raise ValueError(f"score_anneal_frac must be in [0, 1], got {self.score_anneal_frac}")
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {tuple(ARCHS)}, got {self.arch!r}")
        self.hidden = tuple(self.hidden)
        self.betas = tuple(self.betas)

    def model_kwargs(self, pos_norm) -> dict:
        """Constructor kwargs of the backbone this config selects."""
        shared = {"pos_norm": pos_norm, "obs_steps": self.obs_steps, "use_phase": self.use_phase}
        if self.arch == "mlp":
            return {**shared, "hidden": self.hidden, "activation": self.activation}
        if is_mlp(self.arch):  # a resized MLP: its width and depth are the class's own
            return {**shared, "activation": self.activation, **self.net_kwargs}
        return {**shared, **self.net_kwargs}


DEFAULT_BANDWIDTH_PX = 32.0


def is_mlp(arch: str) -> bool:
    """Whether ``arch`` is :class:`PushTMLP` or one of its resized subclasses."""
    return issubclass(ARCHS[arch], PushTMLP)


ARCH_SUFFIX = {"mlp": "", "mlp_medium": "_mlpmed", "mlp_medium_deep": "_mlpdeep",
               "mlp_large": "_mlplarge",
               "transformer": "_tf", "transformer_medium": "_tfmed",
               "transformer_medium_deep": "_tfdeep", "transformer_large": "_tflarge",
               "transformer_big": "_tfbig"}


def method_name(cfg: PolicyConfig) -> str:
    """``mlp``, ``fm`` (MLP+phase), ``score_w<λ>_cos<frac>`` (MLP+phase+score) or
    ``mlp_score_w<λ>_cos<frac>`` (MLP+score, no phase input) -- the skate study's run names --
    with ``_h<px>`` appended when the score bandwidth is not the default (hyperparameter sweeps).
    An unannealed run (``score_anneal_frac = 0``) is ``_const`` in place of ``_cos<frac>``.

    A non-MLP backbone appends :data:`ARCH_SUFFIX`, so ``arch="mlp"`` names -- every run that
    exists today, and every figure and ``output/`` path keyed on them -- are unchanged."""
    if cfg.score_weight > 0:
        schedule = f"cos{cfg.score_anneal_frac:g}" if cfg.score_anneal_frac else "const"
        name = f"score_w{cfg.score_weight:g}_{schedule}"
        if cfg.bandwidth_px != DEFAULT_BANDWIDTH_PX:
            name += f"_h{cfg.bandwidth_px:g}"
        name = name if cfg.use_phase else f"mlp_{name}"
    else:
        name = "fm" if cfg.use_phase else "mlp"
    return name + ARCH_SUFFIX[cfg.arch]


# Figure labels, keyed by run name.
METHODS = {"mlp": "MLP", "fm": "MLP+phase", "score_w0.05_cos0.5": "MLP+phase+score",
           "mlp_tf": "Transformer", "fm_tf": "Transformer+phase",
           "score_w0.05_cos0.5_tf": "Transformer+phase+score"}


def policy_run_dir(root: str | Path, cfg: PolicyConfig) -> Path:
    return Path(root) / "pusht" / "policy" / method_name(cfg) / f"seed{cfg.seed}"


def score_weight_at(cfg: PolicyConfig, step: int, total_steps: int) -> float:
    """``fm_bc.skate.training.cdm_weight_at`` with the cosine schedule, restated (theseus).

    ``score_anneal_frac = 0`` disables the schedule: lambda stays at ``score_weight`` throughout,
    so the regularizer is still acting on the weights the run ends with."""
    if not cfg.score_anneal_frac:
        return cfg.score_weight
    progress = min(step / (cfg.score_anneal_frac * total_steps), 1.0)
    return 0.5 * cfg.score_weight * (1.0 + math.cos(math.pi * progress))


def lr_factor(cfg: PolicyConfig, step: int, total_steps: int) -> float:
    """diffusers' cosine schedule with warmup, as DP configures it (stepped per batch)."""
    if step < cfg.lr_warmup_steps:
        return step / max(1, cfg.lr_warmup_steps)
    progress = (step - cfg.lr_warmup_steps) / max(1, total_steps - cfg.lr_warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))


def policy_rows(
    ep: PushTEpisodes,
    prep: PushTPrepared,
    episode_ids: torch.Tensor,
    obs_steps: int = 1,
    bandwidth_px: float | None = None,
    device: str | torch.device = "cpu",
    normalizer: str = "minmax",
) -> dict[str, torch.Tensor]:
    """One row per step of ``episode_ids``: observation window, phase velocity target, phase.

    The target at step ``k`` is ``u_k = (L - 1) (x_{k+1} - x_k)`` of the agent position in
    normalized units -- ``agent_delta`` is edge-held at the last step, so every row has one.
    ``dphase = 1 / (L - 1)`` is the row's phase increment. With ``bandwidth_px`` also the agent
    score at every row (training rows only need it).
    """
    ep = ep.to(device)
    w = make_windows(ep, to=obs_steps, tp=1, device=device, episode_ids=episode_ids)
    step = w["obs_idx"][:, -1]
    pos = prep.normalizer("agent_pos", normalizer)
    span = (ep.lengths[ep.episode_index[step]] - 1).clamp(min=1).to(torch.float32)  # L - 1
    rows = {
        "obs": w["obs"],  # (R, To, 5) raw
        "velocity": ep.agent_delta[step] / pos.scale.to(span) * span[:, None],  # (R, 2) dx/drho
        "dphase": 1.0 / span,  # (R,)
        "phase": ep.phase[step],  # (R,)
        "episode": ep.episode_index[step],
        "step": step,
    }
    if bandwidth_px is not None:
        grid = pos.apply(prep.grid.agent_positions().to(device)[prep.train_ids.to(device)])
        n_grid = grid.shape[1]
        j = (rows["phase"] * (n_grid - 1)).round().long()  # nearest grid phase
        samples = grid.transpose(0, 1)[j]  # (R, M, 2)
        h = bandwidth_px / float(pos.scale.mean())
        rows["score"] = agent_score(pos.apply(rows["obs"][:, -1, :2]), samples, h)
    return rows


@torch.no_grad()
def _predict(model: Policy, rows: dict[str, torch.Tensor], batch: int = 65536) -> torch.Tensor:
    return torch.cat([model(rows["obs"][lo : lo + batch], rows["phase"][lo : lo + batch])
                      for lo in range(0, rows["obs"].shape[0], batch)])


@torch.no_grad()
def validation_loss(model: Policy, rows: dict[str, torch.Tensor]) -> float:
    """Velocity MSE in normalized units -- the training loss without the regularizer."""
    was_training = model.training
    model.eval()
    loss = float((_predict(model, rows) - rows["velocity"]).pow(2).mean())
    model.train(was_training)
    return loss


@torch.no_grad()
def evaluate_policy_offline(model: Policy, rows: dict[str, torch.Tensor]) -> dict:
    """Phase-velocity error against the demonstrations, overall and per phase tercile.

    ``velocity_rmse`` is in normalized units per unit phase (the loss's units). ``rmse_px`` is the
    same error as the pixel error of the next position after one Euler step with the
    demonstration's ``drho``; it pools steps, ``episode_rmse_px`` averages per-episode RMSEs.
    ``zero_*`` is the trivial policy ``u = 0`` (stand still), for scale.
    """
    was_training = model.training
    model.eval()
    pred = _predict(model, rows)
    scale = model.pos_scale
    verr = (pred - rows["velocity"]).norm(dim=-1)  # (R,) per unit phase
    err = ((pred - rows["velocity"]) * scale).norm(dim=-1) * rows["dphase"]  # (R,) px
    zero = rows["velocity"].norm(dim=-1)
    zero_px = (rows["velocity"] * scale).norm(dim=-1) * rows["dphase"]
    ph = rows["phase"]
    out = {
        "velocity_mse": float((pred - rows["velocity"]).pow(2).mean()),
        "velocity_rmse": float(verr.pow(2).mean().sqrt()),
        "zero_velocity_rmse": float(zero.pow(2).mean().sqrt()),
        "rmse_px": float(err.pow(2).mean().sqrt()),
        "mae_px": float(err.mean()),
        "zero_rmse_px": float(zero_px.pow(2).mean().sqrt()),
    }
    eps = rows["episode"].unique()
    per_ep = torch.stack([err[rows["episode"] == e].pow(2).mean().sqrt() for e in eps])
    out["episode_rmse_px"] = float(per_ep.mean())
    out["n_episodes"], out["n_rows"] = int(eps.numel()), int(err.numel())
    for name, (a, b) in {"t1": (0.0, 1 / 3), "t2": (1 / 3, 2 / 3), "t3": (2 / 3, 1.01)}.items():
        keep = (ph >= a) & (ph < b)
        out[f"velocity_rmse_{name}"] = float(verr[keep].pow(2).mean().sqrt())
    model.train(was_training)
    return out


_LOG_FIELDS = [
    "epoch", "global_step", "seconds", "lr", "score_weight", "train_loss", "train_velocity_loss",
    "train_score_l1", "val_loss", "val_loss_ema", "held_out_velocity_rmse_ema", "is_best",
]


def _write_log(path: Path, log: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(log)


def train_pusht_policy(
    ep: PushTEpisodes,
    prep: PushTPrepared,
    cfg: PolicyConfig,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[Policy, dict]:
    """Train one arm; returns the best (lowest EMA validation loss) model and its evaluation.

    With ``out_dir`` writes ``config.json``, ``log.csv`` (one row per epoch), ``best.pt``,
    ``epoch_XXXX.pt`` every ``checkpoint_every`` epochs, ``last.pt`` and, last, as the
    finished-run marker, ``metrics.json`` -- train / val / held-out scores of both ``best`` and
    ``last``. Every checkpoint holds the EMA weights when ``use_ema``.
    """
    torch.manual_seed(cfg.seed)
    device = cfg.device
    gen = torch.Generator(device=device).manual_seed(cfg.seed)
    held_ids = prep.held_out_ids

    rows = lambda ids, **kw: policy_rows(ep, prep, ids, cfg.obs_steps, device=device,  # noqa: E731
                                         normalizer=cfg.normalizer, **kw)
    train = rows(prep.train_ids, bandwidth_px=cfg.bandwidth_px)
    val, held = rows(prep.val_ids), rows(held_ids)
    model = make_pusht_policy(
        cfg.arch, **cfg.model_kwargs(prep.normalizer("agent_pos", cfg.normalizer))).to(device)
    ema = type(model)(**model.config()).to(device).requires_grad_(False) if cfg.use_ema else model
    if cfg.use_ema:
        ema.load_state_dict(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=cfg.betas, eps=cfg.eps,
                            weight_decay=cfg.weight_decay)

    n_rows = train["obs"].shape[0]
    steps_per_epoch = math.ceil(n_rows / cfg.batch_size)  # DataLoader, drop_last=False
    total_steps = steps_per_epoch * cfg.num_epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(cfg, s, total_steps))

    info = {
        "method": method_name(cfg),
        "train_config": asdict(cfg),
        "pipeline": "Diffusion Policy pusht_lowdim workspace (see fm_bc.pusht.policy_training)",
        "n_params": sum(p.numel() for p in model.parameters()),
        "n_train_rows": n_rows,
        "steps_per_epoch": steps_per_epoch,
        "total_steps": total_steps,
        "train_episodes": prep.train_ids.tolist(),
        "val_episodes": prep.val_ids.tolist(),
        "held_out_episodes": held_ids.tolist(),
        "arch": cfg.arch,
        "inputs": ("[o, x, rho]: block [xy_n, cos, sin], agent xy_n, 2 rho - 1" if is_mlp(cfg.arch)
                   else f"[o, x] as {model.n_context_tokens} cross-attention tokens, "
                        "rho by adaptive layer norm"),
        "phase_source": "oracle t/(L-1) of the demonstration",
        "target": f"agent phase velocity dx/drho = (L-1)(x_{{k+1}} - x_k), {cfg.normalizer}-normalized",
        "selection": "best.pt = EMA model at the lowest EMA validation loss (velocity MSE)",
        "score": {"density": "agent xy at the row's phase, training episodes on the phase grid",
                  "bandwidth_px": cfg.bandwidth_px, "n_grid": prep.grid.n_grid,
                  "mean_norm": float(train["score"].norm(dim=-1).mean())},
    }
    out = Path(out_dir) if out_dir is not None else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(info, indent=2))

    log: list[dict] = []
    best = {"val_loss_ema": math.inf, "epoch": -1, "state": None}
    t0 = time.time()
    global_step = 0
    model.train()
    for epoch in range(cfg.num_epochs):
        perm = torch.randperm(n_rows, device=device, generator=gen)
        sums = torch.zeros(3, device=device)  # total, velocity, |b|
        lam = 0.0
        for lo in range(0, n_rows, cfg.batch_size):
            idx = perm[lo : lo + cfg.batch_size]
            pred = model(train["obs"][idx], train["phase"][idx])
            target = train["velocity"][idx]
            velocity_loss = (pred - target).pow(2).mean()
            b = ((target - pred) * train["score"][idx]).sum(-1).abs().mean()
            lam = score_weight_at(cfg, global_step, total_steps) if cfg.score_weight > 0 else 0.0
            loss = velocity_loss + lam * b if lam > 0 else velocity_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {global_step}: {loss.item()}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            global_step += 1
            if cfg.use_ema:
                _ema_update(ema, model, ema_decay(cfg, global_step))
            sums += torch.stack([loss.detach(), velocity_loss.detach(), b.detach()])
        sums = (sums / steps_per_epoch).tolist()

        row = {"epoch": epoch, "global_step": global_step, "seconds": round(time.time() - t0, 2),
               "lr": sched.get_last_lr()[0], "score_weight": lam, "train_loss": sums[0],
               "train_velocity_loss": sums[1], "train_score_l1": sums[2], "is_best": 0}
        if epoch % cfg.val_every == 0 or epoch == cfg.num_epochs - 1:
            row["val_loss"] = validation_loss(model, val)
            row["val_loss_ema"] = validation_loss(ema, val)
            row["held_out_velocity_rmse_ema"] = evaluate_policy_offline(ema, held)["velocity_rmse"]
            if row["val_loss_ema"] < best["val_loss_ema"]:
                best = {"val_loss_ema": row["val_loss_ema"], "epoch": epoch,
                        "state": {k: v.detach().clone() for k, v in ema.state_dict().items()}}
                row["is_best"] = 1
                if out is not None:
                    ema.save(out / "best.pt", info=info, epoch=epoch, val_loss=row["val_loss_ema"])
        if out is not None and (epoch + 1) % cfg.checkpoint_every == 0:
            ema.save(out / f"epoch_{epoch + 1:04d}.pt", info=info, epoch=epoch)
        log.append(row)
        if verbose and (epoch % 25 == 0 or epoch == cfg.num_epochs - 1):
            print(f"  epoch {epoch:>4}  step {global_step:>6}  λ {lam:.3g}  "
                  f"train {row['train_velocity_loss']:.3f}  val {row.get('val_loss', math.nan):.3f} "
                  f"(ema {row.get('val_loss_ema', math.nan):.3f})  best epoch {best['epoch']}  "
                  f"({row['seconds']:.0f}s)", flush=True)

    def scores(m: Policy) -> dict:
        return {"held_out": evaluate_policy_offline(m, held),
                "val_split": evaluate_policy_offline(m, val),
                "train": evaluate_policy_offline(m, rows(prep.train_ids))}

    final = {"last": scores(ema)}
    if out is not None:
        ema.save(out / "last.pt", info=info, epoch=cfg.num_epochs - 1)
    ema.load_state_dict(best["state"])
    final["best"] = {**scores(ema), "epoch": best["epoch"], "val_loss_ema": best["val_loss_ema"]}
    final.update(info)
    final["train_seconds"] = time.time() - t0
    if out is not None:
        _write_log(out / "log.csv", log)
        (out / "metrics.json").write_text(json.dumps(final, indent=2))
    return ema, final
