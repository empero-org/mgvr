"""The MGVR synthetic grid: experiments x arms x learning rates x seeds, as parallel processes
on one GPU (the models are tiny, so several share it), skipping runs that already finished.

    python -m mgvr.suite --grid cal  --out runs/cal [--jobs 8]
    python -m mgvr.suite --grid main --out runs/main
    python -m mgvr.analyze runs/main

Every arm gets the same learning-rate sweep and the best rate per (experiment, arm) is chosen
by mean held-out accuracy over the run (analyze.py) -- no arm is tuned more than another.

`cal` (calibration, 2026-10-12): the first grid was off both ways -- repeated-token induction at
d128 / batch 64 / lr up to 4.5e-3 saturated within 50 steps in every arm, while MQAR with 64
pairs and two-hop with 40 chains stayed at 6-8% after 3-4k steps (Zoology trains ~30x more
sequences at that difficulty). `cal` brackets slower induction (bigger vocabulary, d64, batch 8,
loss on every position, evals every 10 steps) and smaller recall problems, baseline vs MGVR at
one seed; `main` is set from it (experiments where the baseline needs a fair share of the
budget to reach 90%).
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time

_IND = dict(task="induction", n_layers=2, d_model=64, seq_len=256, vocab=4096, batch=8,
            steps=1500, eval_every=10, loss="all")
_MQ = dict(task="mqar", n_layers=2, d_model=128, vocab=512, batch=64, steps=4000, eval_every=25)
_HOP = dict(task="mqar_2hop", n_layers=4, d_model=128, vocab=512, batch=64, steps=6000,
            eval_every=50)
GRIDS = {
    "cal": {
        "ind_lr1e-3": dict(_IND, lrs=(1e-3,)),
        "ind_lr3e-4": dict(_IND, lrs=(3e-4,)),
        "mqar_n16": dict(_MQ, seq_len=64, lrs=(1e-3,)),
        "mqar_n32": dict(_MQ, seq_len=128, lrs=(1e-3,)),
        "mqar_n32_lr3e-3": dict(_MQ, seq_len=128, lrs=(3e-3,)),
        "hop_n10": dict(_HOP, seq_len=60, lrs=(1e-3,)),
        "hop_n20": dict(_HOP, seq_len=120, lrs=(1e-3,)),
    },
    # Set from `cal` (2026-10-12): induction at vocab 4096 / d64 / batch 8 never formed in
    # 1.5k steps, MQAR stayed at 10-20% in 4k steps, two-hop learned partway -- with MGVR's
    # AUC 26% (10 chains) and 49% (20 chains) above the baseline's at one seed. So: induction
    # in the original setting (where it formed within 50 steps at lr 4.5e-3) at three FIXED
    # learning rates with evals every 5 steps, so formation takes hundreds of steps and its
    # timing is resolved; two-hop at both sizes; MQAR with 16 pairs given 3x the steps.
    "main": {
        **{f"ind2L_lr{lr:g}": dict(task="induction", n_layers=2, d_model=128, seq_len=256,
                                   vocab=512, batch=64, steps=600, eval_every=5, lrs=(lr,))
           for lr in (1e-4, 3e-4, 1e-3)},
        "hop_n10": dict(_HOP, seq_len=60, lrs=(1e-3, 3e-3)),
        "hop_n20": dict(_HOP, seq_len=120, lrs=(1e-3, 3e-3)),
        "mqar_n16": dict(_MQ, seq_len=64, steps=12000, eval_every=100, lrs=(1e-3, 3e-3)),
    },
}
ARMS = ("none", "resformer", "resformer_learn", "mgvr", "mgvr_gate", "mgvr_mix")
LRS = (5e-4, 1.5e-3, 4.5e-3)
SEEDS = (0, 1, 2)
DEFAULTS = dict(vocab=512, batch=64, eval_every=50, loss="mask", n_heads=4)


def run_dir(out: str, exp: str, arm: str, lr: float, seed: int) -> str:
    return os.path.join(out, exp, arm, f"lr{lr:g}", f"s{seed}")


def jobs(out: str, grid: dict, arms, seeds, only=None):
    for exp, e in grid.items():
        if only and exp not in only:
            continue
        e = {**DEFAULTS, **e}
        for arm, lr, seed in itertools.product(arms, e.get("lrs", LRS), seeds):
            d = run_dir(out, exp, arm, lr, seed)
            if os.path.exists(os.path.join(d, "summary.json")):
                continue
            cmd = [sys.executable, "-m", "mgvr.train", "--task", e["task"],
                   "--vres", arm, "--lr", str(lr), "--seed", str(seed), "--steps", str(e["steps"]),
                   "--n-layers", str(e["n_layers"]), "--d-model", str(e["d_model"]),
                   "--n-heads", str(e["n_heads"]), "--seq-len", str(e["seq_len"]),
                   "--vocab", str(e["vocab"]), "--batch", str(e["batch"]),
                   "--eval-every", str(e["eval_every"]), "--loss", e["loss"], "--out", d]
            yield (exp, arm, lr, seed), d, cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", choices=sorted(GRIDS), default="main")
    p.add_argument("--out", default="runs/main")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--arms", nargs="*", default=None)
    p.add_argument("--seeds", nargs="*", type=int, default=None)
    p.add_argument("--only", nargs="*")
    a = p.parse_args()
    arms = a.arms or (("none", "mgvr") if a.grid == "cal" else ARMS)
    seeds = a.seeds or ((0,) if a.grid == "cal" else SEEDS)
    todo = list(jobs(a.out, GRIDS[a.grid], arms, seeds, a.only))
    print(f"{len(todo)} runs to do, {a.jobs} at a time", flush=True)
    running, done, failed, t0 = [], 0, 0, time.time()
    while todo or running:
        while todo and len(running) < a.jobs:
            key, d, cmd = todo.pop(0)
            os.makedirs(d, exist_ok=True)
            fh = open(os.path.join(d, "stdout.txt"), "w")
            running.append((subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh, key))
        time.sleep(0.5)
        for item in list(running):
            proc, fh, key = item
            if proc.poll() is None:
                continue
            fh.close()
            running.remove(item)
            done += 1
            if proc.returncode:
                failed += 1
                print(f"FAILED {key} (exit {proc.returncode})", flush=True)
            if done % 10 == 0 or not (todo or running):
                print(f"{done} done ({failed} failed), {time.time() - t0:.0f} s", flush=True)
    print("SUITE_DONE", flush=True)


if __name__ == "__main__":
    main()
