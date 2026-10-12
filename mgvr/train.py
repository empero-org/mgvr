"""One synthetic run: a TinyLM arm on one task, logging held-out task accuracy over training
and, for `induction`, each head's induction and previous-token attention scores.

    python -m mgvr.train --task induction --vres mgvr --lr 1.5e-3 --seed 0 \\
        --steps 2000 --out runs/main/x

Controls: every arm of a (task, seed) pair sees the same training batches (the generator is
seeded by the seed alone), the same held-out batch (a fixed seed shared by everything) and
the same initial weights (the value-residual parameters draw nothing; model.py). Optimizer:
AdamW, linear warmup then cosine to 10%; weight decay on matrices only -- not on embeddings,
norms, or the value-residual parameters (W_z, L, lam).
Writes <out>/log.jsonl (one line per eval) and <out>/summary.json.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

try:  # `python -m mgvr.train` or a plain script
    from .model import Config, TinyLM
    from .tasks import TASKS, induction_offset
except ImportError:  # pragma: no cover
    from model import Config, TinyLM
    from tasks import TASKS, induction_offset

EVAL_SEED = 2026_10_12


def lr_at(step: int, steps: int, warmup: int, lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / warmup
    p = (step - warmup) / max(1, steps - warmup)
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))


def param_groups(model: TinyLM, wd: float):
    no_decay = set(map(id, model.vres_parameters())) | {id(model.embed.weight)}
    decay, rest = [], []
    for p in model.parameters():
        (decay if p.ndim >= 2 and id(p) not in no_decay else rest).append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": rest, "weight_decay": 0.0}]


@torch.no_grad()
def head_scores(atts, T: int, L: int):
    """Per (layer, head): the mean attention from each repeat position t >= L to t - L + 1 (the
    token after the current token's earlier occurrence: the induction pattern), and from every
    position t >= 1 to t - 1 (the previous-token pattern). atts: per layer [B, H, T, T]."""
    t_ind = torch.arange(L, T)
    t_prev = torch.arange(1, T)
    ind, prev = [], []
    for a in atts:
        ind.append(a[:, :, t_ind, t_ind - L + 1].mean(dim=(0, 2)))
        prev.append(a[:, :, t_prev, t_prev - 1].mean(dim=(0, 2)))
    return torch.stack(ind).tolist(), torch.stack(prev).tolist()


@torch.no_grad()
def evaluate(model, task: str, xe, ye, me, amp, T: int):
    model.eval()
    with amp():
        out = model(xe, return_attn=task == "induction")
    logits, atts = out if isinstance(out, tuple) else (out, None)
    lg, tg = logits[me].float(), ye[me]
    res = {"loss": F.cross_entropy(lg, tg).item(), "acc": (lg.argmax(-1) == tg).float().mean().item()}
    if atts is not None:
        ind, prev = head_scores(atts, T, induction_offset(T))
        res.update(ind_score=ind, prev_score=prev, ind_max=max(map(max, ind)),
                   prev_max=max(map(max, prev)))
    model.train()
    return res


def first_step(log, key, thr):
    for r in log:
        if r.get(key) is not None and r[key] >= thr:
            return r["step"]
    return None


def run(a) -> dict:
    dev = torch.device(a.device if a.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(a.seed)
    cfg = Config(vocab=a.vocab, d_model=a.d_model, n_layers=a.n_layers, n_heads=a.n_heads,
                 seq_len=a.seq_len, vres=a.vres, vres_lambda=a.vres_lambda)
    model = TinyLM(cfg).to(dev)
    opt = torch.optim.AdamW(param_groups(model, a.wd), lr=a.lr, betas=(0.9, 0.95), eps=1e-8)
    gen = torch.Generator(device=dev).manual_seed(1000 + a.seed)          # training data
    egen = torch.Generator(device=dev).manual_seed(EVAL_SEED)             # held-out batch
    task = TASKS[a.task]
    xe, ye, me = task(a.eval_batch, a.seq_len, a.vocab, egen, dev)
    amp = ((lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if dev.type == "cuda"
           else (lambda: torch.autocast("cpu", enabled=False)))
    os.makedirs(a.out, exist_ok=True)
    log, t0 = [], time.time()
    with open(os.path.join(a.out, "log.jsonl"), "w") as f:
        for step in range(a.steps + 1):
            if step % a.eval_every == 0 or step == a.steps:
                r = {"step": step, **evaluate(model, a.task, xe, ye, me, amp, a.seq_len)}
                log.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
            if step == a.steps:
                break
            for g in opt.param_groups:
                g["lr"] = lr_at(step, a.steps, a.warmup, a.lr)
            x, y, m = task(a.batch, a.seq_len, a.vocab, gen, dev)
            with amp():
                logits = model(x)
            if a.loss == "mask":
                loss = F.cross_entropy(logits[m].float(), y[m])
            else:
                loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
            if not torch.isfinite(loss):
                log.append({"step": step, "diverged": True})
                break
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    accs = [r["acc"] for r in log if "acc" in r]
    summary = {
        "args": vars(a), "params": sum(p.numel() for p in model.parameters()),
        "diverged": any(r.get("diverged") for r in log),
        "final_acc": sum(accs[-3:]) / max(1, len(accs[-3:])),
        "best_acc": max(accs) if accs else 0.0,
        "auc": sum(accs) / max(1, len(accs)),          # mean held-out accuracy over the run
        "step_acc50": first_step(log, "acc", 0.5), "step_acc90": first_step(log, "acc", 0.9),
        "seconds": time.time() - t0,
    }
    if a.task == "induction":
        summary.update(step_ind30=first_step(log, "ind_max", 0.3),
                       step_ind50=first_step(log, "ind_max", 0.5),
                       final_ind_max=log[-1].get("ind_max"), final_prev_max=log[-1].get("prev_max"))
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", choices=sorted(TASKS), required=True)
    p.add_argument("--vres", default="none")
    p.add_argument("--vres-lambda", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1.5e-3)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--loss", choices=("mask", "all"), default="mask")
    p.add_argument("--vocab", type=int, default=512)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--device", default="")
    p.add_argument("--out", required=True)
    return p


if __name__ == "__main__":
    s = run(parser().parse_args())
    print(json.dumps({k: v for k, v in s.items() if k != "args"}))
