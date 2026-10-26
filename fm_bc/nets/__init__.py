"""Ported network backbones: Diffusion Policy's 1D UNet, MIP's residual MLP, ACT's CVAE
transformer and an AdaLN transformer.

The first three back the generative baselines; :class:`PhaseTransformer` backs the deterministic
phase-conditioned policies, as an architectural alternative to the MLPs that concatenate the phase
to their input.

:mod:`~fm_bc.nets.act` is the one backbone here that uses ``nn.MultiheadAttention`` rather than
the hand-rolled softmax of :mod:`~fm_bc.nets.transformer`; its module docstring says why.
"""

from .act import ACT, kl_divergence, sinusoidal_pos_embedding
from .mip_mlp import MIPEncoder, MIPResidualMLP
from .transformer import PhaseEmbedding, PhaseTransformer
from .unet1d import ConditionalUnet1D

__all__ = ["ACT", "ConditionalUnet1D", "MIPEncoder", "MIPResidualMLP", "PhaseEmbedding",
           "PhaseTransformer", "kl_divergence", "sinusoidal_pos_embedding"]
