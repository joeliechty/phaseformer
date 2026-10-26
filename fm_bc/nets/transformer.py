"""An AdaLN-conditioned transformer backbone: learned query tokens that read a context by
cross-attention, with a scalar phase driving every normalization.

Ported from ``models/support_models.py`` and ``models/conditional_flow_matching_transformer.py``
of https://github.com/.../flow_matching_transformer (local: ``fmt_project``). The source is a
conditional flow-matching model whose ``phase`` argument is the flow time; here the same machinery
carries the *task* phase ``rho``, which is the point of the port -- the MLP arms
(:class:`fm_bc.skate.policy.FlowMLP`, :class:`fm_bc.pusht.policy.PushTMLP`) inject ``rho`` by
concatenating ``2 rho - 1`` to their input vector, and this injects it multiplicatively at every
LayerNorm instead.

Deviations from the source, all deliberate:

* **The phase embedding uses uniform frequencies**, ``linspace(0, max_freq, dim // 2)``, not the
  source's log-spaced 10000-base sinusoid. The phase is in ``[0, 1]``; over that range almost every
  log-spaced channel is constant to machine precision. This is
  :meth:`fm_bc.nets.mip_mlp.MIPResidualMLP._embed_time`'s embedding, which is in the repo for the
  same reason.
* **adaLN-Zero.** Each modulation is one zero-initialized ``Linear`` emitting ``(shift, scale,
  gate)``, and every residual branch is gated, so a fresh block is exactly the identity. The source
  Xavier-initializes the modulation and has no gate, so its blocks start by scaling and shifting
  their activations at random.
* **No null token and no classifier-free guidance.** They are meaningless for a deterministic
  regressor, and the source's training-time token dropout would make the loss stochastic.
* **Cross-attention is optional** (``use_cross_attn``). The source allocates a full
  ``CrossAttention`` per block even when it is never called, leaving dead parameters in every
  checkpoint.
* **Dropout defaults to 0**, as in the MLP arms; the source defaults to 0.1.

Also dropped, as training-loop coupling that does not belong in a backbone: the ODE integrator
(``sample`` / ``inference``), the loss, the checkpoint helpers, and the 2-D sin/cos image-grid
positional embedding. That removes the source's ``utils.tf_utils`` / ``utils.euclid_utils``
imports, so none of its SE(3) code comes across.

**Attention is the hand-rolled ``(q @ k^T).softmax()``, not**
``torch.nn.functional.scaled_dot_product_attention``. :func:`fm_bc.skate.policy.model_divergence`
differentiates the policy's input Jacobian with ``create_graph=True``, i.e. it backpropagates
through a backward pass; SDPA's fused kernels do not implement double backward, and the explicit
matmul/softmax does.

Like the rest of :mod:`fm_bc.nets`, this imports torch and nothing else.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["PhaseEmbedding", "PhaseTransformer"]


class PhaseEmbedding(nn.Module):
    """``(B,) -> (B, phase_dim)``: uniform-frequency sin/cos, then a 2-layer MLP."""

    def __init__(self, phase_dim: int = 128, max_freq: float = 100.0):
        super().__init__()
        if phase_dim % 2:
            raise ValueError(f"phase_dim must be even, got {phase_dim}")
        self.phase_dim = phase_dim
        self.register_buffer("frequencies", torch.linspace(0, max_freq, phase_dim // 2))
        self.mlp = nn.Sequential(
            nn.Linear(phase_dim, phase_dim), nn.SiLU(), nn.Linear(phase_dim, phase_dim)
        )

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        angles = phase.reshape(-1, 1) * self.frequencies.unsqueeze(0)
        return self.mlp(torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1))


class _Modulation(nn.Module):
    """``(B, phase_dim) -> (shift, scale, gate)``, each ``(B, 1, dim)``. Zero-initialized.

    At init ``shift = scale = gate = 0``, so ``_AdaLN`` is a plain LayerNorm and the gated residual
    branch contributes nothing: the block is the identity (DiT's adaLN-Zero).
    """

    def __init__(self, dim: int, phase_dim: int):
        super().__init__()
        self.proj = nn.Linear(phase_dim, 3 * dim)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)
        self.act = nn.SiLU()

    def forward(self, phase_emb: torch.Tensor) -> tuple[torch.Tensor, ...]:
        shift, scale, gate = self.proj(self.act(phase_emb)).unsqueeze(1).chunk(3, dim=-1)
        return shift, scale, gate


class _AdaLN(nn.Module):
    """Affine-free LayerNorm, then ``x * (1 + scale) + shift``."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return self.norm(x) * (1 + scale) + shift


def _attend(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float,
            dropout: nn.Dropout) -> torch.Tensor:
    """``softmax(q k^T * scale) v`` for ``(b, heads, n, head_dim)`` inputs.

    Fused (``scaled_dot_product_attention``) whenever grad is off, explicit matmul/softmax
    whenever it is on. The explicit form is what makes this module twice differentiable, which
    :func:`fm_bc.skate.policy.model_divergence` needs and the fused kernels do not implement (module
    docstring) -- but it costs several small kernels per block, and at batch 1 that dispatch is
    most of the forward pass. Grad is the right switch rather than ``self.training``: the
    divergence is taken under ``create_graph=True`` on a model that may be in eval mode, while
    rollouts and benchmarks run under ``torch.no_grad``.

    The two paths agree to ~1e-6 in float32, so an inference result does not depend on which
    ran; `tests/test_transformer_attention.py` pins that.
    """
    if torch.is_grad_enabled():
        return dropout(((q @ k.transpose(-2, -1)) * scale).softmax(dim=-1)) @ v
    p = dropout.p if dropout.training else 0.0
    return nn.functional.scaled_dot_product_attention(q, k, v, dropout_p=p, scale=scale)


class _SelfAttention(nn.Module):
    """Multi-head self-attention, written out so it is twice differentiable (module docstring)."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"hidden_dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = _attend(q, k, v, self.scale, self.dropout)
        return self.dropout(self.proj(out.transpose(1, 2).reshape(b, n, c)))


class _CrossAttention(nn.Module):
    """Multi-head cross-attention: queries from ``x``, keys and values from ``context``."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"hidden_dim {dim} must be divisible by num_heads {num_heads}")
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        m = context.shape[1]
        q = self.q(x).reshape(b, n, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        kv = self.kv(context).reshape(b, m, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        out = _attend(q, k, v, self.scale, self.dropout)
        return self.dropout(self.proj(out.transpose(1, 2).reshape(b, n, c)))


class _FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Block(nn.Module):
    """Self-attention, optional cross-attention and an MLP, each adaLN-modulated and gated."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 phase_dim: int = 128, use_cross_attn: bool = True, use_self_attn: bool = True):
        super().__init__()
        if use_self_attn:
            self.norm1, self.mod1 = _AdaLN(dim), _Modulation(dim, phase_dim)
            self.attn = _SelfAttention(dim, num_heads, dropout)
        else:
            self.norm1 = self.mod1 = self.attn = None
        if use_cross_attn:
            self.norm2, self.mod2 = _AdaLN(dim), _Modulation(dim, phase_dim)
            self.cross_attn = _CrossAttention(dim, num_heads, dropout)
        else:
            self.norm2 = self.mod2 = self.cross_attn = None
        self.norm3, self.mod3 = _AdaLN(dim), _Modulation(dim, phase_dim)
        self.mlp = _FeedForward(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, phase_emb: torch.Tensor,
                context: torch.Tensor | None = None) -> torch.Tensor:
        if self.attn is not None:
            shift, scale, gate = self.mod1(phase_emb)
            x = x + gate * self.attn(self.norm1(x, shift, scale))
        if self.cross_attn is not None and context is not None:
            shift, scale, gate = self.mod2(phase_emb)
            x = x + gate * self.cross_attn(self.norm2(x, shift, scale), context)
        shift, scale, gate = self.mod3(phase_emb)
        return x + gate * self.mlp(self.norm3(x, shift, scale))


class PhaseTransformer(nn.Module):
    """``(context (B, M, hidden_dim), phase (B,)) -> (B, n_tokens, out_dim)``.

    The ``n_tokens`` outputs come from learned query tokens that self-attend and then cross-attend
    to ``context``, which the caller has already projected to ``hidden_dim`` -- task-specific
    tokenization stays in the task module. Passing ``in_dim`` instead makes the queries an input:
    ``forward`` then requires ``x (B, n_tokens, in_dim)``, which is what an action-chunk flow model
    would need.

    ``use_phase=False`` drops the phase: the modulation is driven by a constant embedding, so the
    network cannot read ``phase`` and the argument is accepted and ignored. That keeps one call
    signature across the arms, as :class:`fm_bc.pusht.policy.PushTMLP` does.

    The output projection is zero-initialized, so an untrained model predicts exactly zero.

    Note that attention is only doing work when there is more than one token to attend over:
    ``n_tokens = 1`` makes self-attention a no-op and ``M = 1`` makes cross-attention one, in which
    case this is an adaLN-conditioned residual MLP.

    ``use_self_attn=False`` drops the self-attention branch, leaving a gated cross-attention and a
    gated MLP per block. It exists because at ``n_tokens = 1`` self-attention is worse than a
    no-op, it is *dead weight*: a softmax over one key is the constant 1 whatever the query, so
    ``qkv``'s q and k rows receive exactly zero gradient, and what survives,
    ``proj(W_v(adaLN(x)))``, is a linear gated residual branch that the block's own MLP already
    subsumes. Cross-attention is unaffected -- its keys come from the context, so it stays live at
    any ``n_tokens``. The default is ``True``, which builds the module tree this file has always
    built.
    """

    def __init__(self, out_dim: int, n_tokens: int, hidden_dim: int = 128, num_layers: int = 2,
                 num_heads: int = 4, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 phase_dim: int = 128, max_freq: float = 100.0, use_phase: bool = True,
                 use_cross_attn: bool = True, in_dim: int | None = None,
                 use_self_attn: bool = True):
        super().__init__()
        if n_tokens < 1:
            raise ValueError(f"n_tokens must be >= 1, got {n_tokens}")
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {num_layers}")
        self.out_dim, self.n_tokens = int(out_dim), int(n_tokens)
        self.hidden_dim, self.num_layers = int(hidden_dim), int(num_layers)
        self.num_heads, self.mlp_ratio = int(num_heads), float(mlp_ratio)
        self.dropout, self.phase_dim = float(dropout), int(phase_dim)
        self.max_freq, self.use_phase = float(max_freq), bool(use_phase)
        self.use_cross_attn, self.in_dim = bool(use_cross_attn), in_dim
        self.use_self_attn = bool(use_self_attn)

        if in_dim is None:
            self.queries = nn.Parameter(torch.randn(1, self.n_tokens, self.hidden_dim) * 0.02)
            self.input_proj = None
        else:
            self.queries = None
            self.input_proj = nn.Linear(int(in_dim), self.hidden_dim)
        # Position in the query sequence. Constant (and so inert) when there is one token.
        self.query_pos = nn.Parameter(torch.randn(1, self.n_tokens, self.hidden_dim) * 0.02)

        if self.use_phase:
            self.phase_emb = PhaseEmbedding(self.phase_dim, self.max_freq)
        else:
            self.phase_emb = None
            self.register_buffer("_no_phase", torch.zeros(1, self.phase_dim))

        self.blocks = nn.ModuleList([
            _Block(self.hidden_dim, self.num_heads, self.mlp_ratio, self.dropout, self.phase_dim,
                   self.use_cross_attn, self.use_self_attn)
            for _ in range(self.num_layers)
        ])
        self.final_norm, self.final_mod = _AdaLN(self.hidden_dim), _Modulation(self.hidden_dim,
                                                                              self.phase_dim)
        self.output_proj = nn.Linear(self.hidden_dim, self.out_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier on every Linear, then re-zero the adaLN modulations and the output head.

        ``_Modulation`` zero-initializes itself, but ``self.apply`` would overwrite that, so the
        zeroing is repeated here after the sweep.
        """
        def basic(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(basic)
        for module in self.modules():
            if isinstance(module, _Modulation):
                nn.init.zeros_(module.proj.weight)
                nn.init.zeros_(module.proj.bias)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def embed_phase(self, phase: torch.Tensor, batch: int) -> torch.Tensor:
        if not self.use_phase:
            return self._no_phase.expand(batch, -1)
        return self.phase_emb(phase.reshape(-1))

    def tokens(self, batch: int, x: torch.Tensor | None) -> torch.Tensor:
        if self.input_proj is None:
            if x is not None:
                raise ValueError("this backbone uses learned queries; build it with in_dim to "
                                 "pass x")
            h = self.queries.expand(batch, -1, -1)
        else:
            if x is None:
                raise ValueError("this backbone was built with in_dim; x is required")
            if x.shape[1] != self.n_tokens:
                raise ValueError(f"x must have {self.n_tokens} tokens, got {x.shape[1]}")
            h = self.input_proj(x)
        return h + self.query_pos

    def forward(self, context: torch.Tensor | None, phase: torch.Tensor,
                x: torch.Tensor | None = None) -> torch.Tensor:
        batch = context.shape[0] if context is not None else x.shape[0]
        phase_emb = self.embed_phase(phase, batch)
        h = self.tokens(batch, x)
        for block in self.blocks:
            h = block(h, phase_emb, context)
        shift, scale, _ = self.final_mod(phase_emb)
        return self.output_proj(self.final_norm(h, shift, scale))
