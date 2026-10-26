"""Action-chunk baselines: Diffusion Policy, MIP, MeanFlow and ACT.

All four share Diffusion Policy's interface -- the last ``To = 2`` poses in, a chunk of
``Tp = 16`` controls out, ``Ta = 8`` of them executed before re-planning (see
:mod:`fm_bc.skate.chunks`) -- and differ in the model of the chunk. The first three are generative;
ACT is a deterministic regressor and is here as the contrast:

* :class:`DiffusionPolicy` (Chi et al. 2023): DDPM, 100 steps, ``squaredcos_cap_v2``,
  epsilon prediction, ``ConditionalUnet1D``; sampled with the full 100-step DDPM chain.
* :class:`MIPPolicy` ("Much Ado About Noising", the Minimum Iterative Policy): two regression
  steps, the second from the first's output with noise injected at ``t* = 0.9`` in training
  only; MIP's residual MLP. This is the repo's ``mip_loss`` / ``mip_sampler`` (the paper's
  Eq. 4.4 up to the ``t*`` scaling of the second step's input, see the repo's note).
* :class:`MeanFlowPolicy` (Geng et al. 2025): the average velocity ``u(z, r, t)`` trained with
  the MeanFlow identity via a forward-mode JVP, sampled in one step ``x = e - u(e, 0, 1)``;
  the same UNet as Diffusion Policy with a second time input ``t - r``.
* :class:`ACTPolicy` (Zhao et al. 2023): a CVAE transformer trained with L1 + ``kl_weight`` KL,
  whose latent is fixed to the prior mean ``z = 0`` at inference -- so the chunk is a
  deterministic function of the observation window. Ported at LeRobot's defaults
  (:mod:`fm_bc.nets.act`), including its single decoder layer.

Chunks are generated in the normalized action space ``[-1, 1]`` (the control box) and returned
in physical units. Every sampler takes an explicit ``torch.Generator`` so that rollouts are
reproducible.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn
from torch.func import jvp

from skate_planner.planner import SkateLimits

from ..nets import ACT, ConditionalUnet1D, MIPEncoder, MIPResidualMLP, kl_divergence
from .chunks import ACT_DIM, OBS_DIM, embed_poses, normalize_actions, unnormalize_actions

__all__ = ["BASELINES", "ACTPolicy", "ChunkPolicy", "DiffusionPolicy", "MIPPolicy",
           "MeanFlowPolicy", "make_policy", "load_policy", "cosine_betas"]


class ChunkPolicy(nn.Module):
    """``(obs poses (B, To, 3)) -> control chunk (B, Tp, 2)``, sampled."""

    kind: str = ""

    def __init__(self, obs_steps: int = 2, horizon: int = 16, action_steps: int = 8,
                 limits: dict | None = None):
        super().__init__()
        if not 0 < action_steps <= horizon - obs_steps + 1:
            raise ValueError(f"need 0 < action_steps <= horizon - obs_steps + 1, got "
                             f"{action_steps}, {horizon}, {obs_steps}")
        self.obs_steps, self.horizon, self.action_steps = obs_steps, horizon, action_steps
        self.limits = SkateLimits(**(limits or {}))
        self._init_args: dict = {}

    def cond(self, obs: torch.Tensor) -> torch.Tensor:
        """Embedded observation window ``(B, To, 4)``."""
        return embed_poses(obs)

    def loss(self, obs: torch.Tensor, controls: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Training loss on physical control chunks ``(B, Tp, 2)``."""
        return self.loss_normalized(obs, normalize_actions(controls, self.limits))

    def loss_normalized(self, obs, actions) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Physical control chunk ``(B, Tp, 2)``."""
        return unnormalize_actions(self.sample_normalized(obs, generator), self.limits)

    def sample_normalized(self, obs, generator) -> torch.Tensor:
        raise NotImplementedError

    def _noise(self, obs: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        return torch.randn((obs.shape[0], self.horizon, ACT_DIM), generator=generator,
                           device=obs.device, dtype=obs.dtype)

    def config(self) -> dict:
        return {"obs_steps": self.obs_steps, "horizon": self.horizon,
                "action_steps": self.action_steps,
                "limits": {"v_max": self.limits.v_max, "v_min": self.limits.v_min,
                           "omega_max": self.limits.omega_max},
                **self._init_args}

    def save(self, path: str | Path, **extra) -> None:
        torch.save({"kind": self.kind, "config": self.config(), "state_dict": self.state_dict(),
                    **extra}, path)


class DiffusionPolicy(ChunkPolicy):
    kind = "dp"

    def __init__(self, num_train_timesteps: int = 100, down_dims=(256, 512, 1024),
                 kernel_size: int = 5, n_groups: int = 8, diffusion_step_embed_dim: int = 256,
                 **kwargs):
        super().__init__(**kwargs)
        self._init_args = {"num_train_timesteps": num_train_timesteps,
                           "down_dims": list(down_dims), "kernel_size": kernel_size,
                           "n_groups": n_groups,
                           "diffusion_step_embed_dim": diffusion_step_embed_dim}
        self.net = ConditionalUnet1D(ACT_DIM, global_cond_dim=OBS_DIM * self.obs_steps,
                                     diffusion_step_embed_dim=diffusion_step_embed_dim,
                                     down_dims=down_dims, kernel_size=kernel_size,
                                     n_groups=n_groups, cond_predict_scale=True)
        self.num_train_timesteps = num_train_timesteps
        betas = cosine_betas(num_train_timesteps)
        self.register_buffer("alphas_cumprod", torch.cumprod(1.0 - betas, dim=0))

    def eps(self, x: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.net(x, t, global_cond=self.cond(obs).flatten(1))

    def loss_normalized(self, obs, actions):
        noise = torch.randn_like(actions)
        t = torch.randint(0, self.num_train_timesteps, (actions.shape[0],), device=actions.device)
        ac = self.alphas_cumprod[t][:, None, None]
        noisy = ac.sqrt() * actions + (1 - ac).sqrt() * noise
        return ((self.eps(noisy, t, obs) - noise) ** 2).mean(), {}

    def sample_normalized(self, obs, generator):
        """The DDPM reverse chain of diffusers' ``DDPMScheduler.step`` with ``fixed_small``
        variance and ``clip_sample=True``."""
        x = self._noise(obs, generator)
        ac = self.alphas_cumprod
        for t in reversed(range(self.num_train_timesteps)):
            eps = self.eps(x, torch.full((x.shape[0],), t, device=x.device), obs)
            a_t = ac[t]
            a_prev = ac[t - 1] if t > 0 else ac.new_tensor(1.0)
            b_t, b_prev = 1 - a_t, 1 - a_prev
            alpha, beta = a_t / a_prev, 1 - a_t / a_prev
            x0 = ((x - b_t.sqrt() * eps) / a_t.sqrt()).clamp(-1, 1)
            x = (a_prev.sqrt() * beta / b_t) * x0 + (alpha.sqrt() * b_prev / b_t) * x
            if t > 0:
                var = (b_prev / b_t * beta).clamp(min=1e-20)
                x = x + var.sqrt() * self._noise(obs, generator)
        return x


def cosine_betas(n: int, max_beta: float = 0.999) -> torch.Tensor:
    """diffusers' ``squaredcos_cap_v2`` schedule (``betas_for_alpha_bar``), float32."""
    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    return torch.tensor([min(1 - alpha_bar((i + 1) / n) / alpha_bar(i / n), max_beta)
                         for i in range(n)], dtype=torch.float32)


class MIPPolicy(ChunkPolicy):
    kind = "mip"

    def __init__(self, t_two_step: float = 0.9, loss_scale: float = 0.1, emb_dim: int = 512,
                 n_layers: int = 8, timestep_emb_dim: int = 128, dropout: float = 0.1,
                 n_encoder_layers: int = 2, **kwargs):
        super().__init__(**kwargs)
        self._init_args = {"t_two_step": t_two_step, "loss_scale": loss_scale,
                           "emb_dim": emb_dim, "n_layers": n_layers,
                           "timestep_emb_dim": timestep_emb_dim, "dropout": dropout,
                           "n_encoder_layers": n_encoder_layers}
        self.t_two_step, self.loss_scale = t_two_step, loss_scale
        self.encoder = MIPEncoder(OBS_DIM, self.obs_steps, emb_dim, n_encoder_layers)
        self.net = MIPResidualMLP(ACT_DIM, self.horizon, emb_dim, self.obs_steps, emb_dim,
                                  n_layers, timestep_emb_dim, dropout=dropout)

    def _step(self, x, t: float, emb):
        tt = x.new_full((x.shape[0],), t)
        return self.net(x, tt, tt, emb)  # the repo's FlowMap.get_velocity: s = t

    def loss_normalized(self, obs, actions):
        emb = self.encoder(self.cond(obs))
        ts = self.t_two_step
        a_t = actions + (1 - ts) * torch.randn_like(actions)
        pred0 = self._step(torch.zeros_like(actions), 0.0, emb)
        pred1 = self._step(a_t, ts, emb)
        loss0 = (((pred0 - actions) / ts) ** 2).sum(-1)
        loss1 = (((pred1 - actions) / (1 - ts)) ** 2).sum(-1)
        return self.loss_scale * (loss0 + loss1).mean(), {
            "mip_loss0": loss0.mean().item(), "mip_loss1": loss1.mean().item()}

    def sample_normalized(self, obs, generator):
        emb = self.encoder(self.cond(obs))
        a0 = self._step(obs.new_zeros((obs.shape[0], self.horizon, ACT_DIM)), 0.0, emb)
        return self._step(a0, self.t_two_step, emb)


class MeanFlowPolicy(ChunkPolicy):
    """``z_t = (1 - t) x + t e`` (``t = 1`` noise), ``v = e - x``; the network is the average
    velocity ``u(z, r, t)``, conditioned on ``(t, t - r)``. ``time_scale`` maps ``t`` in
    ``[0, 1]`` onto the sinusoidal embedding's range (DiT's convention of 1000)."""

    kind = "meanflow"

    def __init__(self, down_dims=(256, 512, 1024), kernel_size: int = 5, n_groups: int = 8,
                 diffusion_step_embed_dim: int = 256, time_scale: float = 1000.0,
                 ratio_r_neq_t: float = 0.25, logit_mean: float = -0.4, logit_std: float = 1.0,
                 adaptive_p: float = 1.0, adaptive_c: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self._init_args = {"down_dims": list(down_dims), "kernel_size": kernel_size,
                           "n_groups": n_groups,
                           "diffusion_step_embed_dim": diffusion_step_embed_dim,
                           "time_scale": time_scale, "ratio_r_neq_t": ratio_r_neq_t,
                           "logit_mean": logit_mean, "logit_std": logit_std,
                           "adaptive_p": adaptive_p, "adaptive_c": adaptive_c}
        self.net = ConditionalUnet1D(ACT_DIM, global_cond_dim=OBS_DIM * self.obs_steps,
                                     diffusion_step_embed_dim=diffusion_step_embed_dim,
                                     down_dims=down_dims, kernel_size=kernel_size,
                                     n_groups=n_groups, cond_predict_scale=True, n_times=2)
        self.time_scale = time_scale
        self.ratio_r_neq_t = ratio_r_neq_t
        self.logit_mean, self.logit_std = logit_mean, logit_std
        self.adaptive_p, self.adaptive_c = adaptive_p, adaptive_c

    def u(self, z, r, t, cond):
        s = self.time_scale
        return self.net(z, t * s, global_cond=cond, time2=(t - r) * s)

    def sample_times(self, b: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        """``(r, t)``: two logit-normal draws, ``t`` the larger; ``r = t`` for all but a
        ``ratio_r_neq_t`` fraction of the batch."""
        s = torch.sigmoid(torch.randn(2, b, device=device, dtype=dtype) * self.logit_std
                          + self.logit_mean)
        t, r = s.max(dim=0).values, s.min(dim=0).values
        keep = torch.rand(b, device=device) < self.ratio_r_neq_t
        return torch.where(keep, r, t), t

    def loss_normalized(self, obs, actions):
        b = actions.shape[0]
        r, t = self.sample_times(b, actions.device, actions.dtype)
        e = torch.randn_like(actions)
        t3, r3 = t[:, None, None], r[:, None, None]
        z = (1 - t3) * actions + t3 * e
        v = e - actions
        cond = self.cond(obs).flatten(1)
        u, dudt = jvp(lambda z_, r_, t_: self.u(z_, r_, t_, cond), (z, r, t),
                      (v, torch.zeros_like(r), torch.ones_like(t)))
        target = (v - (t3 - r3) * dudt).detach()
        sq = ((u - target) ** 2).flatten(1).sum(-1)
        w = (1.0 / (sq.detach() + self.adaptive_c) ** self.adaptive_p)
        return (w * sq).mean(), {"meanflow_mse": sq.mean().item() / u[0].numel()}

    def sample_normalized(self, obs, generator):
        e = self._noise(obs, generator)
        b = e.shape[0]
        return e - self.u(e, e.new_zeros(b), e.new_ones(b), self.cond(obs).flatten(1))


class ACTPolicy(ChunkPolicy):
    """ACT (Zhao et al. 2023): a CVAE whose decoder regresses the chunk under L1.

    The odd one out among these baselines -- it is **deterministic** at inference, because ACT
    sets the latent to the prior mean ``z = 0`` rather than sampling it. ``generator`` is
    accepted and ignored, and the chunk is a function of the observation window alone. That is
    the point of having it here: it is the chunking baseline that cannot split between modes.

    Dropout (0.1, ACT's default) is active in ``train()`` mode, so sampling is only deterministic
    in ``eval()``. Every path in this repo that samples goes through
    :func:`fm_bc.skate.evaluation.evaluate_policy`, which sets it.
    """

    kind = "act"

    def __init__(self, kl_weight: float = 10.0, hidden_dim: int = 512,
                 dim_feedforward: int = 3200, nheads: int = 8, enc_layers: int = 4,
                 dec_layers: int = 1, vae_enc_layers: int = 4, dropout: float = 0.1,
                 latent_dim: int = 32, pre_norm: bool = False, **kwargs):
        super().__init__(**kwargs)
        self._init_args = {"kl_weight": kl_weight, "hidden_dim": hidden_dim,
                           "dim_feedforward": dim_feedforward, "nheads": nheads,
                           "enc_layers": enc_layers, "dec_layers": dec_layers,
                           "vae_enc_layers": vae_enc_layers, "dropout": dropout,
                           "latent_dim": latent_dim, "pre_norm": pre_norm}
        self.kl_weight, self.latent_dim = kl_weight, latent_dim
        self.net = ACT(OBS_DIM, ACT_DIM, obs_steps=self.obs_steps, horizon=self.horizon,
                       hidden_dim=hidden_dim, dim_feedforward=dim_feedforward, nheads=nheads,
                       enc_layers=enc_layers, dec_layers=dec_layers,
                       vae_enc_layers=vae_enc_layers, dropout=dropout, latent_dim=latent_dim,
                       pre_norm=pre_norm)

    def loss_normalized(self, obs, actions):
        a_hat, mu, logvar = self.net(self.cond(obs), actions)
        l1 = (a_hat - actions).abs().mean()  # ACT's reduction: the mean over every element
        kl = kl_divergence(mu, logvar)
        return l1 + self.kl_weight * kl, {"act_l1": l1.item(), "act_kl": kl.item()}

    def sample_normalized(self, obs, generator):
        return self.net(self.cond(obs))[0]  # z = 0; `generator` is unused by construction


BASELINES = {cls.kind: cls for cls in (DiffusionPolicy, MIPPolicy, MeanFlowPolicy, ACTPolicy)}


def make_policy(kind: str, **kwargs) -> ChunkPolicy:
    if kind not in BASELINES:
        raise ValueError(f"baseline must be one of {tuple(BASELINES)}, got {kind!r}")
    return BASELINES[kind](**kwargs)


def load_policy(path: str | Path, map_location="cpu"):
    """Any saved policy -- a :class:`ChunkPolicy`, a ``FlowMLP`` or a ``FlowTransformer``."""
    ckpt = torch.load(path, map_location=map_location)
    if "kind" not in ckpt:  # FlowMLP checkpoints predate `kind`; its absence is the signal.
        from .policy import FlowMLP
        return FlowMLP.load(path, map_location)
    if ckpt["kind"] == "flow_transformer":
        from .policy import FlowTransformer
        return FlowTransformer.load(path, map_location)
    model = make_policy(ckpt["kind"], **ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt
