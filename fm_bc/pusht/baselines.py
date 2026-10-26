"""Action-chunk baselines for Push-T: Diffusion Policy, MIP and ACT.

The Push-T counterparts of :mod:`fm_bc.skate.baselines` (which imports ``skate_planner`` and so
cannot be used here). Same interface as Diffusion Policy's: the last ``To = 2`` states in, a chunk
of ``Tp = 16`` **absolute pixel targets** out, ``Ta = 8`` of them executed before re-planning
(:mod:`fm_bc.pusht.chunks`). The networks are the repo's ports -- ``ConditionalUnet1D`` for DP,
``MIPEncoder`` + ``MIPResidualMLP`` for MIP, ``ACT`` for ACT -- and the model of the chunk is
unchanged:

* :class:`PushTDiffusionPolicy` (Chi et al. 2023): DDPM, 100 steps, ``squaredcos_cap_v2``,
  epsilon prediction, the full 100-step reverse chain at inference.
* :class:`PushTMIPPolicy` ("Much Ado About Noising"): two regression steps, the second from the
  first's output with noise injected at ``t* = 0.9`` in training only.
* :class:`PushTACTPolicy` (Zhao et al. 2023): a CVAE transformer trained with L1 + ``kl_weight``
  KL, its latent fixed to the prior mean ``z = 0`` at inference -- so, alone among these, the
  chunk is a deterministic function of the state window.

Actions are normalized to ``[-1, 1]`` with the action-space box (``[0, 512]^2``), the ``box`` mode
of :mod:`fm_bc.pusht.normalize`, so the chunk lives in the same range as the skate baselines'
control box. Unlike :class:`fm_bc.pusht.policy.PushTMLP` these predict the environment's action
directly -- the PD target -- so a rollout needs neither a phase clock nor the inverse PD.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from ..nets import ACT, ConditionalUnet1D, MIPEncoder, MIPResidualMLP, kl_divergence
from .chunks import ACT_DIM, OBS_DIM, embed_state
from .normalize import Normalizer

__all__ = ["cosine_betas", "PushTChunkPolicy", "PushTACTPolicy", "PushTDiffusionPolicy",
           "PushTMIPPolicy", "PUSHT_BASELINES", "make_pusht_baseline", "load_pusht_baseline"]


def cosine_betas(n: int, max_beta: float = 0.999) -> torch.Tensor:
    """diffusers' ``squaredcos_cap_v2`` schedule, restated here so this subtree imports without
    ``skate_planner`` (a test pins it to :func:`fm_bc.skate.baselines.cosine_betas`)."""
    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
    return torch.tensor([min(1 - alpha_bar((i + 1) / n) / alpha_bar(i / n), max_beta)
                         for i in range(n)], dtype=torch.float32)


class PushTChunkPolicy(nn.Module):
    """``(state window (B, To, 5)) -> action chunk (B, Tp, 2)`` in pixels."""

    kind: str = ""

    def __init__(self, pos_norm: Normalizer | dict, act_norm: Normalizer | dict,
                 obs_steps: int = 2, horizon: int = 16, action_steps: int = 8):
        super().__init__()
        if not 0 < action_steps <= horizon - obs_steps + 1:
            raise ValueError(f"need 0 < action_steps <= horizon - obs_steps + 1, got "
                             f"{action_steps}, {horizon}, {obs_steps}")
        self.obs_steps, self.horizon, self.action_steps = obs_steps, horizon, action_steps
        for name, norm in (("pos", pos_norm), ("act", act_norm)):
            norm = Normalizer.from_dict(norm) if isinstance(norm, dict) else norm
            self.register_buffer(f"{name}_scale",
                                 torch.as_tensor(norm.scale, dtype=torch.float32).flatten())
            self.register_buffer(f"{name}_offset",
                                 torch.as_tensor(norm.offset, dtype=torch.float32).flatten())
        self._init_args: dict = {}

    @property
    def pos(self) -> Normalizer:
        return Normalizer(scale=self.pos_scale, offset=self.pos_offset)

    @property
    def act(self) -> Normalizer:
        return Normalizer(scale=self.act_scale, offset=self.act_offset)

    def cond(self, obs: torch.Tensor) -> torch.Tensor:
        """Embedded observation window ``(B, To, 6)``."""
        return embed_state(obs, self.pos)

    def loss(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Training loss on physical (pixel) action chunks ``(B, Tp, 2)``."""
        return self.loss_normalized(obs, self.act.apply(actions))

    def loss_normalized(self, obs, actions) -> tuple[torch.Tensor, dict]:
        raise NotImplementedError

    @torch.no_grad()
    def sample(self, obs: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
        """Action chunk in pixels ``(B, Tp, 2)``."""
        return self.act.invert(self.sample_normalized(obs, generator))

    def sample_normalized(self, obs, generator) -> torch.Tensor:
        raise NotImplementedError

    def _noise(self, obs: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
        return torch.randn((obs.shape[0], self.horizon, ACT_DIM), generator=generator,
                           device=obs.device, dtype=obs.dtype)

    def config(self) -> dict:
        return {"pos_norm": {"scale": self.pos_scale.tolist(), "offset": self.pos_offset.tolist()},
                "act_norm": {"scale": self.act_scale.tolist(), "offset": self.act_offset.tolist()},
                "obs_steps": self.obs_steps, "horizon": self.horizon,
                "action_steps": self.action_steps, **self._init_args}

    def save(self, path: str | Path, **extra) -> None:
        torch.save({"kind": self.kind, "config": self.config(), "state_dict": self.state_dict(),
                    **extra}, path)


class PushTDiffusionPolicy(PushTChunkPolicy):
    kind = "pusht_dp"

    def __init__(self, *args, num_train_timesteps: int = 100, down_dims=(256, 512, 1024),
                 kernel_size: int = 5, n_groups: int = 8, diffusion_step_embed_dim: int = 256,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self._init_args = {"num_train_timesteps": num_train_timesteps,
                           "down_dims": list(down_dims), "kernel_size": kernel_size,
                           "n_groups": n_groups,
                           "diffusion_step_embed_dim": diffusion_step_embed_dim}
        self.net = ConditionalUnet1D(ACT_DIM, global_cond_dim=OBS_DIM * self.obs_steps,
                                     diffusion_step_embed_dim=diffusion_step_embed_dim,
                                     down_dims=down_dims, kernel_size=kernel_size,
                                     n_groups=n_groups, cond_predict_scale=True)
        self.num_train_timesteps = num_train_timesteps
        self.register_buffer("alphas_cumprod",
                             torch.cumprod(1.0 - cosine_betas(num_train_timesteps), dim=0))

    def eps(self, x: torch.Tensor, t: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
        return self.net(x, t, global_cond=self.cond(obs).flatten(1))

    def loss_normalized(self, obs, actions, generator: torch.Generator | None = None):
        noise = torch.randn(actions.shape, generator=generator, device=actions.device,
                            dtype=actions.dtype)
        t = torch.randint(0, self.num_train_timesteps, (actions.shape[0],), generator=generator,
                          device=actions.device)
        ac = self.alphas_cumprod[t][:, None, None]
        noisy = ac.sqrt() * actions + (1 - ac).sqrt() * noise
        return ((self.eps(noisy, t, obs) - noise) ** 2).mean(), {}

    def sample_normalized(self, obs, generator):
        """diffusers' ``DDPMScheduler.step``, ``fixed_small`` variance, ``clip_sample=True``."""
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


class PushTMIPPolicy(PushTChunkPolicy):
    kind = "pusht_mip"

    def __init__(self, *args, t_two_step: float = 0.9, loss_scale: float = 0.1,
                 emb_dim: int = 512, n_layers: int = 8, timestep_emb_dim: int = 128,
                 dropout: float = 0.1, n_encoder_layers: int = 2, **kwargs):
        super().__init__(*args, **kwargs)
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
        return self.net(x, tt, tt, emb)

    def loss_normalized(self, obs, actions, generator: torch.Generator | None = None):
        emb = self.encoder(self.cond(obs))
        ts = self.t_two_step
        noise = torch.randn(actions.shape, generator=generator, device=actions.device,
                            dtype=actions.dtype)
        a_t = actions + (1 - ts) * noise
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


class PushTACTPolicy(PushTChunkPolicy):
    """ACT (Zhao et al. 2023) on Push-T: a CVAE whose decoder regresses the chunk under L1.

    Deterministic at inference -- ACT fixes the latent to the prior mean ``z = 0`` rather than
    sampling it, so ``generator`` is accepted and ignored by :meth:`sample_normalized`. It is the
    chunking baseline that cannot split between modes, which is why it is here.

    ``loss_normalized`` *does* honour ``generator``: it drives the CVAE reparameterization, and
    :func:`fm_bc.pusht.baseline_training.validation_loss` passes a fixed one so that
    ``val_loss_ema`` -- and therefore which epoch becomes ``best.pt`` -- is a deterministic
    function of the weights.
    """

    kind = "pusht_act"

    def __init__(self, *args, kl_weight: float = 10.0, hidden_dim: int = 512,
                 dim_feedforward: int = 3200, nheads: int = 8, enc_layers: int = 4,
                 dec_layers: int = 1, vae_enc_layers: int = 4, dropout: float = 0.1,
                 latent_dim: int = 32, pre_norm: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
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

    def loss_normalized(self, obs, actions, generator: torch.Generator | None = None):
        a_hat, mu, logvar = self.net(self.cond(obs), actions, generator=generator)
        l1 = (a_hat - actions).abs().mean()  # ACT's reduction: the mean over every element
        kl = kl_divergence(mu, logvar)
        return l1 + self.kl_weight * kl, {"act_l1": l1.item(), "act_kl": kl.item()}

    def sample_normalized(self, obs, generator):
        return self.net(self.cond(obs))[0]  # z = 0; `generator` is unused by construction


PUSHT_BASELINES = {cls.kind.removeprefix("pusht_"): cls
                   for cls in (PushTDiffusionPolicy, PushTMIPPolicy, PushTACTPolicy)}


def make_pusht_baseline(kind: str, **kwargs) -> PushTChunkPolicy:
    if kind not in PUSHT_BASELINES:
        raise ValueError(f"baseline must be one of {tuple(PUSHT_BASELINES)}, got {kind!r}")
    return PUSHT_BASELINES[kind](**kwargs)


def load_pusht_baseline(path: str | Path, map_location="cpu") -> tuple[PushTChunkPolicy, dict]:
    ckpt = torch.load(path, map_location=map_location)
    kinds = {cls.kind: cls for cls in (PushTDiffusionPolicy, PushTMIPPolicy, PushTACTPolicy)}
    if ckpt.get("kind") not in kinds:
        raise ValueError(f"not a Push-T chunk policy checkpoint: kind={ckpt.get('kind')!r}")
    model = kinds[ckpt["kind"]](**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, {k: v for k, v in ckpt.items() if k not in ("state_dict", "config")}
