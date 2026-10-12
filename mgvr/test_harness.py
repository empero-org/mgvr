#!/usr/bin/env python
"""Checks for the MGVR synthetic harness (CPU, about a minute):

  - formula: ValueResidual(mgvr) == the definition, one (token, head) at a time, with random
    W_z and L; the gate-only and mix-only ablations likewise;
  - init: every MGVR variant computes resformer's logits bitwise at init (W_z = 0, L = lam I),
    and at lam = 0 the baseline's; resformer_learn starts as resformer; all arms share every
    other weight for a seed (common random numbers);
  - the analysis path (explicit softmax, attention probs) gives the training path's logits;
    rows of the probs sum to 1 and never look ahead; the model is causal;
  - tasks: every scored target is determined by the context (induction repeats, MQAR values,
    two-hop c's), masks and shapes line up, BOS never scored;
  - train: a short run per task logs and summarizes; an induction run reports head scores.

    python -m mgvr.test_harness
"""
from __future__ import annotations

import math
import sys
import tempfile
from dataclasses import replace

import torch

try:
    from .model import VRES, Config, TinyLM, ValueResidual
    from .tasks import TASKS, induction_offset
    from . import train as T
except ImportError:  # pragma: no cover
    from model import VRES, Config, TinyLM, ValueResidual
    from tasks import TASKS, induction_offset
    import train as T

torch.set_num_threads(2)
RESULTS = []


def check(name, cond, detail=""):
    RESULTS.append((name, bool(cond)))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""))


BASE = Config(vocab=64, d_model=32, n_layers=3, n_heads=4, seq_len=24)


def _logits(cfg, x, seed=0, mutate=None):
    torch.manual_seed(seed)
    m = TinyLM(cfg).double()
    if mutate:
        mutate(m)
    return m(x)


def test_formula():
    print("formula")
    g = torch.Generator().manual_seed(1)
    B, Tn, H, dh, d = 2, 5, 4, 8, 32
    x = torch.randn(B, Tn, d, generator=g, dtype=torch.float64)
    v = torch.randn(B, Tn, H, dh, generator=g, dtype=torch.float64)
    v0 = torch.randn(B, Tn, H, dh, generator=g, dtype=torch.float64)
    for kind in ("mgvr", "mgvr_gate", "mgvr_mix"):
        vr = ValueResidual(replace(BASE, d_model=d, n_heads=H, vres=kind)).double()
        with torch.no_grad():
            vr.w_z.normal_(0, 1, generator=g)
            if hasattr(vr, "L"):
                vr.L.add_(0.3 * torch.randn(H, H, generator=g, dtype=torch.float64))
        out = vr(x, v, v0)
        ref = torch.empty_like(v)
        for b in range(B):
            for s in range(Tn):
                z = (vr.w_z @ x[b, s]) / math.sqrt(d)
                gz = 2 * torch.sigmoid(z[:H]) if vr.gate else torch.ones(H, dtype=z.dtype)
                zm = z[H:] if vr.gate else z
                for gi in range(H):
                    acc = gz[gi] * v[b, s, gi]
                    for h in range(H):
                        w = (vr.L[gi, h] + torch.tanh(zm[gi * H + h])) if vr.mix else \
                            (vr.lam if h == gi else 0.0)
                        acc = acc + w * v0[b, s, h]
                    ref[b, s, gi] = acc
        check(f"{kind} == the definition per (token, head)", torch.allclose(out, ref, atol=1e-12),
              f"max err {(out - ref).abs().max().item():.1e}")


def test_init():
    print("init")
    x = torch.randint(1, 64, (2, 24), generator=torch.Generator().manual_seed(2))
    res = _logits(replace(BASE, vres="resformer"), x)
    same = all(torch.equal(_logits(replace(BASE, vres=k), x), res)
               for k in ("resformer_learn", "mgvr", "mgvr_gate", "mgvr_mix"))
    check("every value-residual arm computes resformer's logits at init, bitwise", same)
    none = _logits(replace(BASE, vres="none"), x)
    zero = all(torch.equal(_logits(replace(BASE, vres=k, vres_lambda=0.0), x), none)
               for k in ("resformer", "mgvr", "mgvr_mix", "mgvr_gate"))
    check("at lam = 0 they compute the baseline's, bitwise", zero)
    check("the residual matters: resformer != none at lam 0.5", not torch.equal(res, none))
    torch.manual_seed(0)
    a = TinyLM(replace(BASE, vres="none"))
    torch.manual_seed(0)
    b = TinyLM(replace(BASE, vres="mgvr"))
    shared = dict(a.named_parameters())
    check("common random numbers: every shared weight identical across arms",
          all(torch.equal(p, shared[n]) for n, p in b.named_parameters() if n in shared)
          and set(shared) <= set(dict(b.named_parameters())))


def test_attention_paths():
    print("attention paths and causality")
    cfg = replace(BASE, vres="mgvr")
    torch.manual_seed(3)
    m = TinyLM(cfg).double()
    with torch.no_grad():
        for blk in m.blocks[1:]:
            blk.attn.vres.w_z.normal_(0, 1)
    x = torch.randint(1, 64, (2, 24), generator=torch.Generator().manual_seed(4))
    fast = m(x)
    slow, atts = m(x, return_attn=True)
    check("analysis path (explicit softmax) == training path", torch.allclose(fast, slow, atol=1e-10),
          f"max err {(fast - slow).abs().max().item():.1e}")
    check("attention rows sum to 1 and never look ahead",
          all(torch.allclose(a.sum(-1), torch.ones_like(a.sum(-1))) and
              a.triu(1).abs().max().item() == 0 for a in atts))
    x2 = x.clone()
    x2[:, 15:] = torch.randint(1, 64, x2[:, 15:].shape, generator=torch.Generator().manual_seed(5))
    check("causal: changing tokens 15.. leaves logits 0..14 unchanged",
          torch.allclose(m(x2)[:, :15], fast[:, :15], atol=1e-12, rtol=0))


def test_tasks():
    print("tasks")
    g = torch.Generator().manual_seed(6)
    B, Tn, V = 8, 256, 512
    x, y, m = TASKS["induction"](B, Tn, V, g, "cpu")
    L = induction_offset(Tn)
    t = torch.arange(L, Tn)
    check("induction: each scored target follows the current token's earlier occurrence",
          torch.equal(x[:, t], x[:, t - L]) and torch.equal(y[:, t], x[:, t - L + 1])
          and int(m.sum()) == B * (Tn - L) and torch.equal(x[:, 1:], y[:, :-1]))
    for name, hops in (("mqar", 1), ("mqar_2hop", 2)):
        Tn2 = 256 if name == "mqar" else 240
        x, y, m = TASKS[name](B, Tn2, V, g, "cpu")
        ok = torch.equal(x[:, 1:], y[:, :-1]) and not bool(m[:, 0].any())
        n = Tn2 // 4 if name == "mqar" else Tn2 // 6
        ok = ok and int(m.sum()) == B * n
        for b in range(B):
            ctx_len = 2 * n if name == "mqar" else 4 * n
            pairs = x[b, 1:1 + ctx_len].view(-1, 2).tolist()
            look = {}
            for k, v in pairs:
                ok = ok and k not in look                 # keys (and middle hops) unique
                look[k] = v
            for tpos in m[b].nonzero().squeeze(1).tolist():
                ans = look[int(x[b, tpos])]
                if hops == 2:
                    ans = look[ans]
                ok = ok and ans == int(y[b, tpos])
        check(f"{name}: every scored target is the {hops}-hop lookup of its query; keys unique",
              ok)


def test_train():
    print("train (short CPU runs)")
    with tempfile.TemporaryDirectory() as tmp:
        for task, Tn in (("induction", 64), ("mqar", 64), ("mqar_2hop", 60)):
            a = T.parser().parse_args(["--task", task, "--vres", "mgvr", "--steps", "40",
                                       "--eval-every", "20", "--batch", "8", "--eval-batch", "16",
                                       "--seq-len", str(Tn), "--vocab", "128", "--d-model", "32",
                                       "--n-layers", "2", "--device", "cpu", "--out",
                                       f"{tmp}/{task}"])
            s = T.run(a)
            ok = (not s["diverged"]) and 0 <= s["auc"] <= 1 and s["params"] > 0
            if task == "induction":
                ok = ok and s["final_ind_max"] is not None and 0 <= s["final_ind_max"] <= 1
            check(f"{task}: 40 steps run, log and summary written", ok,
                  f"auc {s['auc']:.3f}, loss logged {len(open(f'{tmp}/{task}/log.jsonl').readlines())} evals")


if __name__ == "__main__":
    for t in (test_formula, test_init, test_attention_paths, test_tasks, test_train):
        t()
    n = sum(ok for _, ok in RESULTS)
    print(f"\n{n}/{len(RESULTS)} passed")
    sys.exit(0 if n == len(RESULTS) else 1)
