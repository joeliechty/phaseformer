"""Train the SE(2) flow-matching policy, with or without a divergence regularizer.

``L = L_FM + lambda * R`` with ``L_FM = E |u - v|^2`` over the controls ``[v, omega]``
(phase-conditioned BC) and ``R`` built from two per-sample terms:

* ``a = div u - div v``, the kNN divergence target minus the model's SE(2) divergence, and
* ``b = (u - v) . grad log p``, the twist error against the kernel score.

``cdm_terms`` picks ``R``: ``"both"`` is conditional divergence matching, ``E |a + b|`` (the L1
form of ``divergence_matching/utils/loss_utils.py``); ``"div"`` (``E |a|``) and ``"score"``
(``E |b|``) are its two terms on their own. Both twists have a zero lateral slot, so ``b`` only
ever sees the score's forward and heading rows.

``cdm_schedule="cosine"`` anneals lambda from ``cdm_weight`` to 0 over the first
``cdm_anneal_frac`` of the steps -- robomimic's ``cosine_reg_schedule`` with a zero floor -- and
trains pure FM from there on.

Units line up without any rescaling: twists in m/s and rad/s, divergences in 1/s, the score
in 1/m and 1/rad, so ``(u - v) . score`` is also in 1/s. (robomimic divides by its 0.05 s
control period only because its actions are pose deltas.)

``cdm_weight = 0`` is the FM baseline. It still logs both terms every ``log_every`` steps,
so every arm reports the same diagnostics. ``use_phase=False`` with ``cdm_weight = 0`` is the
plain MLP baseline: the same regression loss on the same rows, but the network never sees the
phase, so it is a state-feedback behaviour-cloning policy rather than a flow.

``horizon > 1`` (FM only, no regularizer) is FM with action chunking: the same network and
recipe, but from ``(pose_k, phase)`` it regresses the controls ``u_k .. u_{k+horizon-1}``
(edge-padded, :func:`fm_bc.skate.chunks.window_indices` with one observed pose) and a rollout
executes ``action_steps`` of them before re-planning. The diagnostics are logged on the first
control, which is the one FM predicts.

``horizon > 1`` with ``cdm_terms="score"`` and ``cdm_weight > 0`` is the score regularizer
carried along the chunk (``chunk_score_w<λ>``). The predicted chunk is integrated open loop
from ``pose_k`` -- the rollout's exact arc, saturated as a rollout would be -- giving
``x̂^(j)``, ``j = 0 .. horizon-1`` (``x̂^(0) = pose_k``), and each control is scored at its own
state against the demos at its own phase:

    R = mean_j E |(u_{k+j} - v^(j)) . grad log p_{(k+j)/N}(x̂^(j))|

with ``u_{k+j}`` the demo control the chunk regresses to (edge-padded), and the score the same
kernel estimate as ``ds.score`` (:func:`fm_bc.divergence.score_kernel_at`) against the training
demos' poses at step ``min(k + j, N)``. The mean over ``j`` (``λ_j = λ / horizon``) gives each
control the weight FM+score gives its single one, as the MSE is also a mean over the chunk.
``x̂`` is detached: the score is a fixed direction per control, as the stored one is for FM.
"""

from __future__ import annotations

import csv
import json
import math
import re
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch

from ..divergence import score_kernel_at
from ..se2 import se2_exp_retract
from .chunks import window_indices
from .dataset import TrajectoryDataset, add_div_and_score
from .evaluation import evaluate_policy
from .policy import ARCHS, make_flow_policy, model_divergence
from skate_planner.planner import SkateLimits
from skate_planner.skate_model import body_twist

__all__ = [
    "BASELINE_METHODS",
    "CHUNK_METHOD",
    "CDM_TERMS",
    "CDM_SCHEDULES",
    "TrainConfig",
    "stratified_subset",
    "prepare_training_set",
    "make_rows",
    "cdm_weight_at",
    "cdm_residuals",
    "cdm_regularizer",
    "open_loop_poses",
    "chunk_score_residuals",
    "method_name",
    "parse_method",
    "run_dir",
    "train",
]

_CONTROL_ROWS = [0, 2]  # [v, omega] inside the body twist [v, 0, omega]
# Run-directory names of the generative baselines (fm_bc.skate.baselines,
# fm_bc.skate.baseline_training).
BASELINE_METHODS = ("dp", "mip", "meanflow", "act")
CDM_TERMS = ("both", "div", "score")
# Run-directory name of FM with action chunking, and the chunk it is trained with (Diffusion
# Policy's and MIP's horizon; parse_method reports it).
CHUNK_METHOD = "fm_chunk"
CHUNK_HORIZON = 16
# Run-directory prefix of a regularized chunk arm (only the score term is wired).
_CHUNK_PREFIX = "chunk_"
CDM_SCHEDULES = ("constant", "cosine")
# Run-name suffix of a chunk policy re-evaluated with temporal ensembling.
_TE_SUFFIX = "_te"
# Run-name suffix per backbone (fm_bc.skate.policy.ARCHS). The MLP's is empty, so every name that
# exists today is unchanged. Applied inside _TE_SUFFIX, i.e. "fm_tf" and "fm_tf_te0.01".
ARCH_SUFFIX = {"mlp": "", "transformer": "_tf"}
# Run-directory prefix per regularizer; "cdm" keeps the constant-lambda runs' original names.
_TERM_PREFIX = {"both": "cdm", "div": "div", "score": "score"}


@dataclass
class TrainConfig:
    cdm_weight: float = 0.0
    cdm_terms: str = "both"  # regularizer: "both" E|a + b| (CDM), "div" E|a|, "score" E|b|
    cdm_schedule: str = "constant"  # "cosine": anneal cdm_weight to 0, see cdm_weight_at
    cdm_anneal_frac: float = 0.5  # fraction of the steps the cosine takes to reach 0
    steps: int = 20000
    batch_size: int = 2048
    lr: float = 1e-3
    arch: str = "mlp"  # backbone: fm_bc.skate.policy.ARCHS
    hidden: tuple = (256, 256, 256)  # arch="mlp" only
    activation: str = "silu"  # arch="mlp" only
    # Backbone kwargs for a non-MLP arch (hidden_dim, num_layers, num_heads, dropout, ...).
    net_kwargs: dict = field(default_factory=dict)
    use_phase: bool = True  # False: state-only network, no flow-time / phase input
    horizon: int = 1  # > 1: predict an action chunk of this many controls (FM only)
    action_steps: int = 1  # controls of each chunk executed before re-planning
    div_estimator: str = "exact"
    n_probes: int = 1
    n_train: int | None = None  # None: every demonstration
    k: int = 16  # kNN rank for the divergence target
    eval_every: int = 1000
    log_every: int = 100
    mmd_bandwidth: float = 0.3
    clamp_controls: bool = True  # saturate rollouts to the planner's SkateLimits
    seed: int = 0
    device: str = "cuda"
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.cdm_terms not in CDM_TERMS:
            raise ValueError(f"cdm_terms must be one of {CDM_TERMS}, got {self.cdm_terms!r}")
        if self.cdm_schedule not in CDM_SCHEDULES:
            raise ValueError(
                f"cdm_schedule must be one of {CDM_SCHEDULES}, got {self.cdm_schedule!r}")
        if not 0 < self.cdm_anneal_frac <= 1:
            raise ValueError(f"cdm_anneal_frac must be in (0, 1], got {self.cdm_anneal_frac}")
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {tuple(ARCHS)}, got {self.arch!r}")
        if not 0 < self.action_steps <= self.horizon:
            raise ValueError(f"need 0 < action_steps <= horizon, got {self.action_steps}, "
                             f"{self.horizon}")
        if self.horizon > 1 and (not self.use_phase
                                 or (self.cdm_weight > 0 and self.cdm_terms != "score")):
            raise ValueError("action chunking is only wired for FM, alone or with the score "
                             "regularizer: phase input, cdm_weight 0 or cdm_terms 'score'")


def stratified_subset(ds: TrajectoryDataset, n: int, rng: np.random.Generator) -> TrajectoryDataset:
    """``n`` demonstrations split evenly across modes, with divergence and score dropped (they
    must be re-estimated from the subset alone)."""
    modes = np.unique(ds.mode)
    per = n // len(modes)
    idx = np.sort(np.concatenate([
        rng.choice(np.flatnonzero(ds.mode == m), per, replace=False) for m in modes
    ]))
    return replace(
        ds,
        poses=ds.poses[idx], controls=ds.controls[idx], twists=ds.twists[idx],
        mode=ds.mode[idx], starts=ds.starts[idx], goals=ds.goals[idx],
        divergence=None, divergence_xy=None, score=None, div_config={},
    )


def prepare_training_set(ds: TrajectoryDataset, cfg: TrainConfig) -> TrajectoryDataset:
    """Subset to ``cfg.n_train`` and make sure divergence / score come from that subset with
    ``cfg.k``; the stored targets are reused only when they already match."""
    if cfg.n_train is not None and cfg.n_train < len(ds):
        ds = stratified_subset(ds, cfg.n_train, np.random.default_rng(cfg.seed))
    stored = ds.div_config
    if ds.divergence is None or stored.get("method") != "knn" or stored.get("k") != cfg.k:
        ds = add_div_and_score(ds, method="knn", k=cfg.k, device=cfg.device)
    return ds


def make_rows(ds: TrajectoryDataset, device: str, horizon: int = 1) -> dict[str, torch.Tensor]:
    """Flatten ``(pose_k, phase k/N, twist_k, div_k, score_k)`` for ``k = 0..N-1``; with
    ``horizon > 1`` also the edge-padded chunk ``u_k .. u_{k+horizon-1}``."""
    m, n = ds.controls.shape[:2]
    rows = {
        "poses": ds.poses[:, :n],
        "phase": np.broadcast_to(np.arange(n) / n, (m, n)),
        "controls": ds.controls,
        "twist": ds.twists,
        "div": ds.divergence[:, :n],
        "score": ds.score[:, :n],
    }
    if horizon > 1:
        rows["chunk"] = ds.controls[:, window_indices(np.arange(n), n, 1, horizon)[1]]
    return {
        key: torch.as_tensor(np.ascontiguousarray(v), dtype=torch.float32, device=device)
        .reshape(m * n, *v.shape[2:])
        for key, v in rows.items()
    }


def cdm_weight_at(cfg: TrainConfig, step: int) -> float:
    """Regularizer weight at ``step`` (1-based, as in :func:`train`).

    ``"cosine"`` is robomimic's ``cosine_reg_schedule`` with a zero floor, counted in steps:
    ``w/2 (1 + cos(pi min(step / (frac * steps), 1)))``, so ``w`` at the start and exactly 0
    from ``frac * steps`` on.
    """
    if cfg.cdm_schedule == "constant":
        return cfg.cdm_weight
    progress = min(step / (cfg.cdm_anneal_frac * cfg.steps), 1.0)
    return 0.5 * cfg.cdm_weight * (1.0 + math.cos(math.pi * progress))


def cdm_residuals(
    pred_twist: torch.Tensor,
    target_twist: torch.Tensor,
    div_v: torch.Tensor | None,
    div_u: torch.Tensor,
    score: torch.Tensor,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    """Per-sample ``a = div u - div v`` and ``b = (u - v) . score``, ``(B,)`` each.

    ``a`` is ``None`` without ``div_v``: the score-only regularizer never needs the model's
    divergence.
    """
    a = None if div_v is None else div_u - div_v
    return a, ((target_twist - pred_twist) * score).sum(dim=-1)


def cdm_regularizer(a: torch.Tensor | None, b: torch.Tensor, terms: str) -> torch.Tensor:
    """``E |a + b|`` (``"both"``, CDM), ``E |a|`` (``"div"``) or ``E |b|`` (``"score"``)."""
    if terms == "both":
        return (a + b).abs().mean()
    if terms == "div":
        return a.abs().mean()
    if terms == "score":
        return b.abs().mean()
    raise ValueError(f"terms must be one of {CDM_TERMS}, got {terms!r}")


def open_loop_poses(start: torch.Tensor, controls: torch.Tensor, dt: float,
                    limits: SkateLimits | None = None) -> torch.Tensor:
    """States ``x̂^(0) .. x̂^(H-1)``, ``(B, H, 3)``, at which each of the ``(B, H, 2)`` controls
    is executed when the chunk runs open loop from ``(B, 3)`` ``start`` -- saturated to
    ``limits`` and held for ``dt``, exactly as :func:`fm_bc.skate.evaluation.rollout_policy`
    does."""
    pose, out = start, []
    for j in range(controls.shape[1]):
        out.append(pose)
        u = controls[:, j] if limits is None else limits.clamp(controls[:, j])
        pose = se2_exp_retract(pose, dt * body_twist(u))
    return torch.stack(out, dim=1)


def chunk_score_residuals(
    pred_chunk: torch.Tensor,
    target_chunk: torch.Tensor,
    poses: torch.Tensor,
    step: torch.Tensor,
    demo_poses: torch.Tensor,
    dt: float,
    bandwidth: float,
    rot_scale: float = 1.0,
    limits: SkateLimits | None = None,
) -> torch.Tensor:
    """Per-sample, per-control ``b_j = (u_{k+j} - v^(j)) . grad log p_{k+j}(x̂^(j))``, ``(B, H)``.

    ``pred_chunk`` / ``target_chunk`` are ``(B, H, 2)`` controls from ``(B, 3)`` ``poses`` at
    integer ``step`` ``k`` ``(B,)``; ``demo_poses`` is the training demos' ``(N+1, M, 3)``
    poses per step, the sample set of each phase marginal. ``x̂`` is detached (see the module
    docstring).
    """
    b, h = pred_chunk.shape[:2]
    with torch.no_grad():
        states = open_loop_poses(poses, pred_chunk, dt, limits)  # (B, H, 3)
        idx = (step[:, None] + torch.arange(h, device=step.device)).clamp(
            max=demo_poses.shape[0] - 1)
        score = score_kernel_at(states.reshape(b * h, 3), demo_poses[idx.reshape(-1)],
                                bandwidth, rot_scale=rot_scale, chunk=4096).reshape(b, h, 3)
    return ((body_twist(target_chunk) - body_twist(pred_chunk)) * score).sum(dim=-1)


def _cdm_diagnostics(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Batch means of every regularizer, and the two terms' signed means."""
    return {
        "cdm_loss": (a + b).abs().mean().item(),
        "div_l1": a.abs().mean().item(),
        "score_l1": b.abs().mean().item(),
        "divergence_diff": a.mean().item(),
        "velocity_score_dot": b.mean().item(),
    }


def method_name(cdm_weight: float, use_phase: bool = True, cdm_terms: str = "both",
                cdm_schedule: str = "constant", cdm_anneal_frac: float = 0.5,
                baseline: str | None = None, horizon: int = 1,
                temporal_ensemble: float | None = None, arch: str = "mlp") -> str:
    """Run-directory name of an arm.

    ``fm`` / ``mlp`` without a regularizer; otherwise ``{cdm|div|score}_w<λ>``, with
    ``_cos<frac>`` for the cosine schedule and an ``mlp_`` prefix for the state-only network.
    The constant-λ CDM names (``cdm_w0.05``, ``mlp_cdm_w0.05``) are unchanged from before the
    other regularizers existed. A generative baseline (``baseline``, one of
    :data:`BASELINE_METHODS`) is named after itself, and FM with action chunking
    (``horizon > 1``) is ``fm_chunk``, and with the score regularizer ``chunk_score_w<λ>...``.

    A chunk policy re-evaluated with temporal ensembling (``temporal_ensemble = m``, see
    :mod:`fm_bc.skate.evaluation`) is its own name plus ``_te<m>``, e.g. ``dp_te0.01``.

    A non-MLP backbone (``arch``, see :data:`fm_bc.skate.policy.ARCHS`) appends :data:`ARCH_SUFFIX`
    before any ``_te<m>``: ``fm_tf``, ``fm_tf_te0.01``. ``arch="mlp"`` adds nothing, so every name
    that exists today is unchanged. A generative baseline is named after itself and ignores
    ``arch``, which does not apply to it.
    """
    name = _method_name(cdm_weight, use_phase, cdm_terms, cdm_schedule, cdm_anneal_frac,
                        baseline, horizon)
    if baseline is None:
        if arch not in ARCH_SUFFIX:
            raise ValueError(f"arch must be one of {tuple(ARCH_SUFFIX)}, got {arch!r}")
        name += ARCH_SUFFIX[arch]
    if temporal_ensemble is not None:
        name += f"{_TE_SUFFIX}{temporal_ensemble:g}"
    return name


def _method_name(cdm_weight, use_phase, cdm_terms, cdm_schedule, cdm_anneal_frac, baseline,
                 horizon) -> str:
    if baseline is not None:
        return baseline
    if horizon > 1 and cdm_weight == 0:
        return CHUNK_METHOD
    if cdm_weight == 0:
        return "fm" if use_phase else "mlp"
    name = f"{_TERM_PREFIX[cdm_terms]}_w{cdm_weight:g}"
    if cdm_schedule == "cosine":
        name += f"_cos{cdm_anneal_frac:g}"
    if horizon > 1:
        return _CHUNK_PREFIX + name
    return name if use_phase else f"mlp_{name}"


def parse_method(name: str) -> dict:
    """Inverse of :func:`method_name`: its keyword arguments for the arm ``name``.

    A generative baseline reads as an unregularized, phase-free arm with ``baseline`` set;
    every other arm has ``baseline=None``. ``horizon`` is 1 for every arm but ``fm_chunk`` and
    the ``chunk_`` arms. ``temporal_ensemble`` is ``m`` for a ``_te<m>`` re-evaluation, else None.
    ``arch`` is the backbone the :data:`ARCH_SUFFIX` names, ``"mlp"`` when there is no suffix.

    The suffixes are peeled in the reverse of the order :func:`method_name` appends them --
    ``_te<m>`` first, then ``_tf`` -- so both round-trip together.
    """
    te = re.fullmatch(rf"(.+){_TE_SUFFIX}([^_]+)", name)
    ensemble = None
    if te is not None:
        name, ensemble = te[1], float(te[2])
    arch = "mlp"
    for candidate, suffix in ARCH_SUFFIX.items():
        if suffix and name.endswith(suffix):
            name, arch = name[: -len(suffix)], candidate
            break
    return {**_parse_method(name), "temporal_ensemble": ensemble, "arch": arch}


def _parse_method(name: str) -> dict:
    defaults = TrainConfig()
    if name == CHUNK_METHOD:
        return {"cdm_weight": 0.0, "use_phase": True, "cdm_terms": defaults.cdm_terms,
                "cdm_schedule": defaults.cdm_schedule,
                "cdm_anneal_frac": defaults.cdm_anneal_frac, "baseline": None,
                "horizon": CHUNK_HORIZON}
    if name in ("fm", "mlp") or name in BASELINE_METHODS:
        return {"cdm_weight": 0.0, "use_phase": name == "fm", "cdm_terms": defaults.cdm_terms,
                "cdm_schedule": defaults.cdm_schedule,
                "cdm_anneal_frac": defaults.cdm_anneal_frac,
                "baseline": name if name in BASELINE_METHODS else None, "horizon": 1}
    m = re.fullmatch(r"(mlp_|chunk_)?(cdm|div|score)_w([^_]+)(?:_cos([^_]+))?", name)
    if m is None:
        raise ValueError(f"not a method name: {name!r}")
    return {
        "cdm_weight": float(m[3]),
        "use_phase": m[1] != "mlp_",
        "cdm_terms": {p: t for t, p in _TERM_PREFIX.items()}[m[2]],
        "cdm_schedule": "constant" if m[4] is None else "cosine",
        "cdm_anneal_frac": defaults.cdm_anneal_frac if m[4] is None else float(m[4]),
        "baseline": None,
        "horizon": CHUNK_HORIZON if m[1] == _CHUNK_PREFIX else 1,
    }


def run_dir(root: str | Path, scenario: str, n_train: int, cfg: TrainConfig) -> Path:
    method = method_name(cfg.cdm_weight, cfg.use_phase, cfg.cdm_terms, cfg.cdm_schedule,
                         cfg.cdm_anneal_frac, horizon=cfg.horizon, arch=cfg.arch)
    return Path(root) / scenario / f"n{n_train}" / method / f"seed{cfg.seed}"


_LOG_FIELDS = [
    "step", "seconds", "lr", "cdm_weight", "total_loss", "velocity_loss", "cdm_loss", "div_l1",
    "score_l1", "score_chunk_l1", "divergence_diff", "velocity_score_dot", "mmd_mean", "mmd_final",
    "collision_rate", "success_rate", "mode_tv", "saturation_rate",
]


def _write_log(path: Path, log: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_LOG_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(log)


def train(
    ds_train: TrajectoryDataset,
    ds_test: TrajectoryDataset,
    cfg: TrainConfig,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> tuple[FlowMLP, dict]:
    """Train one policy; returns it and its final (full) evaluation.

    ``ds_train`` must already carry divergence and score (see :func:`prepare_training_set`).
    With ``out_dir``, writes ``config.json``, ``log.csv``, ``final.pt``, ``metrics.json`` and
    ``rollouts.npz``.
    """
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    device = cfg.device
    model = make_flow_policy(cfg.arch, cfg.use_phase, cfg.horizon, cfg.action_steps,
                             cfg.hidden, cfg.activation, **cfg.net_kwargs).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps, eta_min=cfg.lr * 0.01)
    rows = make_rows(ds_train, device, cfg.horizon)
    chunked = cfg.horizon > 1
    if chunked:
        # Integer step k of each row, and the demos' poses per step for the chunk score.
        n_steps = ds_train.controls.shape[1]
        rows["step"] = (rows["phase"] * n_steps).round()
        demo_poses = torch.as_tensor(ds_train.poses.transpose(1, 0, 2), dtype=torch.float32,
                                     device=device)
        limits = ds_train.limits if cfg.clamp_controls else None
    n_rows = rows["poses"].shape[0]

    out = Path(out_dir) if out_dir is not None else None
    info = {
        "train_config": asdict(cfg),
        "scenario": ds_train.scenario,
        "n_train": len(ds_train),
        "n_test": len(ds_test),
        "div_config": ds_train.div_config,
    }
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps(info, indent=2))

    def evaluate(full: bool) -> tuple[dict, dict]:
        return evaluate_policy(model, ds_test, bandwidth=cfg.mmd_bandwidth, k=cfg.k,
                               full=full, clamp=cfg.clamp_controls, device=device)

    log: list[dict] = []
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        idx = torch.randint(n_rows, (cfg.batch_size,), device=device, generator=gen)
        batch = {key: v[idx] for key, v in rows.items()}
        lam = cdm_weight_at(cfg, step)
        want_log = step % cfg.log_every == 0 or step == 1
        # Only the divergence terms backpropagate through div v -- the score term needs just the
        # twist -- so the double backward runs only while one of them carries weight.
        div_grad = lam > 0 and cfg.cdm_terms != "score"
        if chunked:
            # The whole chunk is regressed; the diagnostics only look at its first control.
            if want_log:
                twist, div_v = model_divergence(
                    model, batch["poses"], batch["phase"], cfg.div_estimator, cfg.n_probes,
                    create_graph=False,
                )
            pred_chunk = model(batch["poses"], batch["phase"])
            velocity_loss = ((pred_chunk - batch["chunk"]) ** 2).mean()
        elif div_grad or want_log:
            twist, div_v = model_divergence(
                model, batch["poses"], batch["phase"], cfg.div_estimator, cfg.n_probes,
                create_graph=div_grad,
            )
        else:
            twist, div_v = model.twist(batch["poses"], batch["phase"]), None
        if not chunked:
            velocity_loss = ((twist[:, _CONTROL_ROWS] - batch["controls"]) ** 2).mean()
        loss = velocity_loss
        metrics: dict = {}
        if chunked and (lam > 0 or want_log):
            b_chunk = chunk_score_residuals(
                pred_chunk, batch["chunk"], batch["poses"], batch["step"].long(), demo_poses,
                ds_train.dt, ds_train.div_config["bandwidth"],
                ds_train.div_config.get("rot_scale", 1.0), limits,
            ).abs().mean()
            if lam > 0:
                loss = loss + lam * b_chunk
        if want_log or (lam > 0 and not chunked):
            a, b = cdm_residuals(twist, batch["twist"], div_v, batch["div"], batch["score"])
            if lam > 0 and not chunked:
                loss = loss + lam * cdm_regularizer(a, b, cfg.cdm_terms)
            if want_log:
                metrics = _cdm_diagnostics(a, b)
                if chunked:
                    metrics["score_chunk_l1"] = b_chunk.item()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}: {loss.item()}")

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if want_log:
            row = {
                "step": step, "seconds": round(time.time() - t0, 2),
                "lr": sched.get_last_lr()[0], "cdm_weight": lam, "total_loss": loss.item(),
                "velocity_loss": velocity_loss.item(), **metrics,
            }
            if step % cfg.eval_every == 0 or step == cfg.steps:
                ev, _ = evaluate(full=False)
                row.update({key: ev[key] for key in
                            ("mmd_mean", "mmd_final", "collision_rate", "success_rate", "mode_tv",
                             "saturation_rate")})
                if verbose:
                    print(f"  step {step:>6}  λ {lam:.3g}  fm {row['velocity_loss']:.4f}  "
                          f"cdm {row['cdm_loss']:.3f}  div {row['div_l1']:.3f}  "
                          f"score {row['score_l1']:.3f}  mmd {ev['mmd_mean']:.4f}  "
                          f"coll {ev['collision_rate']:.3f}  succ {ev['success_rate']:.3f}  "
                          f"modes {[round(f, 2) for f in ev['mode_fractions']]}  "
                          f"({row['seconds']:.0f}s)")
            log.append(row)

    final, arrays = evaluate(full=True)
    final.update(info)
    final["train_seconds"] = time.time() - t0
    if out is not None:
        _write_log(out / "log.csv", log)
        model.save(out / "final.pt", info=info)
        np.savez_compressed(out / "rollouts.npz", **arrays)
        # Last: metrics.json marks a finished run (run_experiments skips on it).
        (out / "metrics.json").write_text(json.dumps(final, indent=2))
    return model, final
