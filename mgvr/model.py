"""A minimal decoder-only transformer for the MGVR study.

Self-contained on purpose: pre-norm RMSNorm blocks, rotary position
embeddings, causal multi-head attention, a GELU MLP, untied embeddings. The only thing that
differs between arms is how layers >= 1 build their attention values (ValueResidual):

  none             v' = v
  resformer        v' = v + lam * v0                      (Zhou et al. 2024, lam = 0.5 fixed)
  resformer_learn  v' = v + lam_l * v0                    (one learnable scalar per layer)
  mgvr             v'[s,g] = 2 sigmoid(z[s,g]) v[s,g] + sum_h (L[g,h] + tanh(z[s,g,h])) v0[s,h]
  mgvr_gate        v'[s,g] = 2 sigmoid(z[s,g]) v[s,g] + lam * v0[s,g]          (the gate alone)
  mgvr_mix         v'[s,g] = v[s,g] + sum_h (L[g,h] + tanh(z[s,g,h])) v0[s,h]  (the mix alone)

z = x_s W_z / sqrt(d) is read from the layer's normalized input x_s; W_z starts at zero and
L at lam * I, so every MGVR variant starts exactly as `resformer` (and at lam = 0 exactly as
`none`). v0 is layer 0's value projection. Here every query head has its own value head
(H = G); with grouped-query attention the gate and the mix act on the G key-value heads.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

VRES = ("none", "resformer", "resformer_learn", "mgvr", "mgvr_gate", "mgvr_mix")


def _wide(t: torch.Tensor) -> torch.Tensor:
    """FP32 for BF16/FP16 activations; FP32 and float64 (the tests) as they are."""
    return t if t.dtype in (torch.float32, torch.float64) else t.float()


@dataclass
class Config:
    vocab: int = 512
    d_model: int = 128
    n_layers: int = 2
    n_heads: int = 4
    seq_len: int = 256
    mlp_mult: int = 4
    vres: str = "none"
    vres_lambda: float = 0.5
    rope_base: float = 10000.0

    def __post_init__(self):
        if self.vres not in VRES:
            raise ValueError(f"vres must be one of {VRES}, got {self.vres!r}")
        if self.d_model % self.n_heads or (self.d_model // self.n_heads) % 2:
            raise ValueError("d_model / n_heads must be an even integer (rotary halves)")
        if self.vres != "none" and self.n_layers < 2:
            raise ValueError("a value residual needs n_layers >= 2 (layer 0 is the anchor)")


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = _wide(x)
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def rope_tables(T: int, dh: int, base: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=device, dtype=torch.float32) / dh))
    f = torch.outer(torch.arange(T, device=device, dtype=torch.float32), inv)   # [T, dh/2]
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    """x [B, T, H, dh], rotate-half convention."""
    h = x.shape[-1] // 2
    x1, x2 = _wide(x[..., :h]), _wide(x[..., h:])
    c, s = cos[None, :, None, :], sin[None, :, None, :]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1).to(x.dtype)


class ValueResidual(nn.Module):
    """(x, v, v0) -> v' for one layer >= 1; see the module docstring."""

    def __init__(self, cfg: Config):
        super().__init__()
        H, d, kind, lam = cfg.n_heads, cfg.d_model, cfg.vres, float(cfg.vres_lambda)
        self.kind, self.H, self.scale = kind, H, 1.0 / math.sqrt(d)
        self.gate = kind in ("mgvr", "mgvr_gate")
        self.mix = kind in ("mgvr", "mgvr_mix")
        if kind == "resformer_learn":
            self.lam = nn.Parameter(torch.tensor(lam))
        elif kind in ("resformer", "mgvr_gate"):
            self.register_buffer("lam", torch.tensor(lam))
        if self.gate or self.mix:
            n = (H if self.gate else 0) + (H * H if self.mix else 0)
            self.w_z = nn.Parameter(torch.zeros(n, d))       # zero: starts as resformer
        if self.mix:
            self.L = nn.Parameter(torch.eye(H) * lam)

    def forward(self, x, v, v0):
        """x [B, T, d] the layer's normalized input; v, v0 [B, T, H, dh]. FP32 inside."""
        vf, v0f = _wide(v), _wide(v0)
        if self.kind in ("resformer", "resformer_learn"):
            return (vf + self.lam * v0f).to(v.dtype)
        xw = _wide(x)
        z = F.linear(xw, self.w_z.to(xw.dtype)) * self.scale              # [B, T, n]
        H = self.H
        if self.gate:
            out = (2.0 * torch.sigmoid(z[..., :H])).unsqueeze(-1) * vf
            zm = z[..., H:]
        else:
            out, zm = vf, z
        if self.mix:
            M = self.L.to(z.dtype) + torch.tanh(zm.unflatten(-1, (H, H)))   # [B, T, g, h]
            out = out + torch.einsum("btgh,bthd->btgd", M, v0f)
        else:
            out = out + self.lam * v0f
        return out.to(v.dtype)


class Attention(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        d, H = cfg.d_model, cfg.n_heads
        self.H, self.dh = H, d // H
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.vres = ValueResidual(cfg) if cfg.vres != "none" and layer_idx > 0 else None

    def forward(self, x, cos, sin, v0=None, return_attn=False):
        """Returns (output, the anchor v0 for later layers, attention probs or None)."""
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.dh).unbind(2)
        anchor = _wide(v) if v0 is None else v0            # the anchor stays FP32
        if self.vres is not None:
            v = self.vres(x, v, v0)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))                     # [B, H, T, dh]
        att = None
        if return_attn:
            s = (_wide(q) @ _wide(k).transpose(-1, -2)) / math.sqrt(self.dh)
            causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
            att = s.masked_fill(~causal, float("-inf")).softmax(-1)
            y = (att @ _wide(v)).to(v.dtype)
        else:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, d)), anchor, att


class Block(nn.Module):
    def __init__(self, cfg: Config, layer_idx: int):
        super().__init__()
        self.n1, self.n2 = RMSNorm(cfg.d_model), RMSNorm(cfg.d_model)
        self.attn = Attention(cfg, layer_idx)
        self.up = nn.Linear(cfg.d_model, cfg.mlp_mult * cfg.d_model, bias=False)
        self.down = nn.Linear(cfg.mlp_mult * cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin, v0=None, return_attn=False):
        a, v0, att = self.attn(self.n1(x), cos, sin, v0, return_attn)
        x = x + a
        x = x + self.down(F.gelu(self.up(self.n2(x))))
        return x, v0, att


class TinyLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=False)
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, std: float = 0.02):
        """GPT-2 style: N(0, std), residual-branch outputs scaled by 1/sqrt(2 L). Only
        nn.Linear / nn.Embedding weights draw, in module order -- the value-residual
        parameters (W_z = 0, L = lam I, lam) draw nothing, so every arm shares every other
        weight for a given seed (common random numbers)."""
        out_std = std / math.sqrt(2 * self.cfg.n_layers)
        for name, m in self.named_modules():
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=std)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=out_std if name.endswith(("proj", "down")) else std)

    def forward(self, idx, return_attn=False):
        """idx [B, T] -> logits [B, T, vocab] (and the per-layer attention probs)."""
        B, T = idx.shape
        cos, sin = rope_tables(T, self.cfg.d_model // self.cfg.n_heads, self.cfg.rope_base,
                               idx.device)
        x, v0, atts = self.embed(idx), None, []
        for blk in self.blocks:
            x, v0, att = blk(x, cos, sin, v0, return_attn)
            atts.append(att)
        logits = self.head(self.norm(x))
        return (logits, atts) if return_attn else logits

    def vres_parameters(self):
        return [p for n, p in self.named_parameters() if ".vres." in n]
