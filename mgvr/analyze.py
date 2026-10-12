"""Tables and plots from a suite directory (suite.py). No torch needed.

    python -m mgvr.analyze runs/main

For each (experiment, arm) the learning rate with the best mean held-out accuracy over the run
(AUC, averaged over seeds) is kept; the table reports mean +- std over seeds at that rate:
AUC, final accuracy, the step held-out accuracy first reaches 50% / 90% (n/3 seeds that got
there), and for induction the step the best head's induction score first reaches 0.3 / 0.5.
Writes <dir>/summary.md, <dir>/summary.json and <dir>/plots/*.png (if matplotlib is present).
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from collections import defaultdict

try:
    from .suite import ARMS
except ImportError:  # pragma: no cover
    from suite import ARMS


def _ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0), len(xs)


def _fmt(m, s, n=None, total=None, digits=3):
    if m is None:
        return "--" if total is None else f"-- (0/{total})"
    txt = f"{m:.{digits}f} ± {s:.{digits}f}" if digits else f"{m:.0f} ± {s:.0f}"
    return txt if total is None else f"{txt} ({n}/{total})"


def load(root):
    runs = defaultdict(list)           # (exp, arm, lr) -> [(summary, log)]
    for f in glob.glob(os.path.join(root, "*", "*", "lr*", "s*", "summary.json")):
        exp, arm, lrd, _ = f.split(os.sep)[-5:-1]
        s = json.load(open(f))
        log = [json.loads(l) for l in open(os.path.join(os.path.dirname(f), "log.jsonl"))]
        runs[(exp, arm, float(lrd[2:]))].append((s, log))
    return runs


def _experiments(runs):
    """Experiment name -> its config (from a run's args), in directory order."""
    out = {}
    for (exp, _, _), v in sorted(runs.items()):
        if exp not in out:
            a = v[0][0]["args"]
            out[exp] = {k: a[k] for k in ("task", "n_layers", "d_model", "seq_len", "vocab",
                                          "batch", "steps", "loss") if k in a}
    return out


def main(root):
    runs = load(root)
    EXPERIMENTS = _experiments(runs)
    out, md = {}, ["# MGVR synthetic suite\n",
                   "Best learning rate per (experiment, arm) by mean AUC (held-out accuracy averaged "
                   "over the run); mean ± std over seeds. Steps: the first eval at or past the "
                   "threshold (seeds that got there / all).\n"]
    for exp in EXPERIMENTS:
        rows, best = [], {}
        for arm in ARMS:
            cands = {lr: v for (e, a, lr), v in runs.items() if e == exp and a == arm}
            if not cands:
                continue
            lr = max(cands, key=lambda k: statistics.mean(s["auc"] for s, _ in cands[k]))
            best[arm] = (lr, cands[lr])
        if not best:
            continue
        ind = EXPERIMENTS[exp]["task"] == "induction"
        md.append(f"\n## {exp}  ({EXPERIMENTS[exp]})\n")
        head = "| arm | lr | AUC | final acc | step acc≥50% | step acc≥90% |"
        sep = "|---|---|---|---|---|---|"
        if ind:
            head += " step ind≥0.3 | step ind≥0.5 | final ind max |"
            sep += "---|---|---|"
        md += [head, sep]
        out[exp] = {}
        for arm, (lr, v) in best.items():
            S = [s for s, _ in v]
            n = len(S)
            r = {"lr": lr, "n": n, **{k: _ms([s.get(k) for s in S]) for k in
                 ("auc", "final_acc", "step_acc50", "step_acc90", "step_ind30", "step_ind50",
                  "final_ind_max")}}
            out[exp][arm] = r
            line = (f"| {arm} | {lr:g} | {_fmt(*r['auc'][:2])} | {_fmt(*r['final_acc'][:2])} | "
                    f"{_fmt(*r['step_acc50'], n, 0)} | {_fmt(*r['step_acc90'], n, 0)} |")
            if ind:
                line += (f" {_fmt(*r['step_ind30'], n, 0)} | {_fmt(*r['step_ind50'], n, 0)} | "
                         f"{_fmt(*r['final_ind_max'][:2])} |")
            md.append(line)
        _plot(root, exp, best, ind)
    open(os.path.join(root, "summary.md"), "w").write("\n".join(md) + "\n")
    json.dump(out, open(os.path.join(root, "summary.json"), "w"), indent=1)
    print("\n".join(md))


def _plot(root, exp, best, ind):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    os.makedirs(os.path.join(root, "plots"), exist_ok=True)
    keys = [("acc", "held-out task accuracy")] + ([("ind_max", "best head's induction score")]
                                                  if ind else [])
    fig, axes = plt.subplots(1, len(keys), figsize=(6 * len(keys), 4), squeeze=False)
    for ax, (k, title) in zip(axes[0], keys):
        for arm, (lr, v) in best.items():
            steps = [r["step"] for r in v[0][1] if k in r]
            ys = [[r[k] for r in log if k in r] for _, log in v]
            n = min(map(len, ys))
            mean = [statistics.mean(y[i] for y in ys) for i in range(n)]
            sd = [statistics.stdev(y[i] for y in ys) if len(ys) > 1 else 0.0 for i in range(n)]
            ax.plot(steps[:n], mean, label=f"{arm} (lr {lr:g})")
            ax.fill_between(steps[:n], [m - s for m, s in zip(mean, sd)],
                            [m + s for m, s in zip(mean, sd)], alpha=0.15)
        ax.set_title(f"{exp}: {title}")
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "plots", f"{exp}.png"), dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/main")
