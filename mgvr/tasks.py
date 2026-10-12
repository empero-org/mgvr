"""Synthetic tasks for the MGVR study, generated on the fly on the training device.

Each generator returns (x, y, mask), all [B, T]: inputs, next-token targets (y[:, t] =
seq[:, t + 1]) and the scored positions -- where the target is determined by the context, so
task loss and accuracy are measured there (and training, by default, supervises only there).
Token 0 is a BOS / separator and never scored.

  induction  repeated random tokens (Olsson et al. 2022, "In-context learning and induction
             heads"): L random tokens, then the same L again. Scored: the repeat, predicting the
             token after the current one's earlier occurrence -- the induction-head circuit
             (a previous-token head, then a head that matches on it and copies).
  mqar       multi-query associative recall (Arora et al. 2023, Zoology): n key-value pairs
             (distinct keys from [1, V/2), values from [V/2, V)), then the n keys again in a
             random order, each followed by its value. Scored: the n values.
  mqar_2hop  two-hop recall: n chains a -> b -> c given as 2n shuffled pairs (a b) and (b c),
             a, b, c from disjoint thirds of the vocabulary (a and b distinct); then the n a's
             again in a random order, each followed by its c. Scored: the n c's -- two lookups
             composed across layers.
"""
from __future__ import annotations

import torch


def _distinct(B: int, n: int, lo: int, hi: int, gen, device) -> torch.Tensor:
    """[B, n] distinct integers from [lo, hi) per row."""
    if n > hi - lo:
        raise ValueError(f"cannot draw {n} distinct values from [{lo}, {hi})")
    return torch.argsort(torch.rand(B, hi - lo, generator=gen, device=device), dim=1)[:, :n] + lo


def _split(seq: torch.Tensor, T: int):
    return seq[:, :T].contiguous(), seq[:, 1:T + 1].contiguous()


def induction(B: int, T: int, V: int, gen, device):
    L = T // 2 + 1                                       # 2L >= T + 1 tokens
    first = torch.randint(1, V, (B, L), generator=gen, device=device)
    x, y = _split(torch.cat([first, first], 1), T)
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    mask[:, L:] = True              # y[t] = first[t + 1 - L]: after seeing first[t - L] again
    return x, y, mask


def mqar(B: int, T: int, V: int, gen, device):
    n = T // 4                                           # BOS + 2n context + 2n queries = T + 1
    half = V // 2
    keys = _distinct(B, n, 1, half, gen, device)
    vals = torch.randint(half, V, (B, n), generator=gen, device=device)
    perm = torch.argsort(torch.rand(B, n, generator=gen, device=device), dim=1)
    ctx = torch.stack([keys, vals], -1).reshape(B, 2 * n)
    qs = torch.stack([keys.gather(1, perm), vals.gather(1, perm)], -1).reshape(B, 2 * n)
    bos = torch.zeros(B, 1, dtype=torch.long, device=device)
    seq = torch.cat([bos, ctx, qs], 1)
    pad = torch.randint(1, V, (B, T + 1 - seq.shape[1]), generator=gen, device=device)
    x, y = _split(torch.cat([seq, pad], 1), T)
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    mask[:, 1 + 2 * n::2][:, :n] = True                  # at each query key: predict its value
    return x, y, mask


def mqar_2hop(B: int, T: int, V: int, gen, device):
    n = T // 6                                           # BOS + 4n pair tokens + 2n queries
    third = (V - 1) // 3
    a = _distinct(B, n, 1, 1 + third, gen, device)
    b = _distinct(B, n, 1 + third, 1 + 2 * third, gen, device)
    c = torch.randint(1 + 2 * third, V, (B, n), generator=gen, device=device)
    pairs = torch.cat([torch.stack([a, b], -1), torch.stack([b, c], -1)], 1)   # [B, 2n, 2]
    order = torch.argsort(torch.rand(B, 2 * n, generator=gen, device=device), dim=1)
    ctx = pairs.gather(1, order.unsqueeze(-1).expand(-1, -1, 2)).reshape(B, 4 * n)
    perm = torch.argsort(torch.rand(B, n, generator=gen, device=device), dim=1)
    qs = torch.stack([a.gather(1, perm), c.gather(1, perm)], -1).reshape(B, 2 * n)
    bos = torch.zeros(B, 1, dtype=torch.long, device=device)
    seq = torch.cat([bos, ctx, qs], 1)
    pad = torch.randint(1, V, (B, T + 1 - seq.shape[1]), generator=gen, device=device)
    x, y = _split(torch.cat([seq, pad], 1), T)
    mask = torch.zeros(B, T, dtype=torch.bool, device=device)
    mask[:, 1 + 4 * n::2][:, :n] = True                  # at each query a: predict its c
    return x, y, mask


TASKS = {"induction": induction, "mqar": mqar, "mqar_2hop": mqar_2hop}


def induction_offset(T: int) -> int:
    """The repeat length L of `induction` at length T (for the attention scores)."""
    return T // 2 + 1
