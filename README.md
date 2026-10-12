# MGVR: Multi-head Gated Value Residuals

Reference implementation and synthetic study for the Empero AI technical report

> **Identity on Demand: Multi-head Gated Value Residuals** ([PDF](paper.pdf))
> Leon Lehmann, Empero AI Research, October 2026.

A Transformer layer that needs a token's identity, to copy it or to look something up with it, has to
recover it from a residual stream that every earlier layer has written into. Value residuals (ResFormer)
hand every layer a fixed share of the first layer's values. MGVR makes that share a decision. Every layer
computes it from its own input, separately for every token and every key–value head:

```
z_s        = W_z x̂_s / sqrt(d)                       # G gate logits u_s, G×G mixing logits M_s
ṽ[s, g]    = 2·sigmoid(u[s, g])·v[s, g]  +  Σ_h (Λ[g, h] + tanh M[s, g, h])·v0[s, h]
```

Here `v0` is layer 0's values (the anchor) and `Λ` is a static G×G matrix per layer. Attention then runs
unchanged on `ṽ`. With `W_z = 0` and `Λ = λI`, MGVR is exactly ResFormer (`v + λ v0`), and with `λ = 0` it
is exactly standard attention.

**Results.**
- **Pretraining** (controlled comparison over 1B tokens): the same held-out loss in 16–22% fewer steps,
  worth 1.15–1.23× effective compute after MGVR's 4.3% step-time cost. Final loss 0.050 nats lower.
- **Induction heads**: 40% less time from the induction attention pattern to accurate copying.
- **Two-hop associative recall**: learnt 25–49% faster than a Transformer and 7–18% faster than ResFormer.
- **Cost**: 0.2% of the active parameters. The key–value cache size is unchanged.

## Use it

`mgvr/model.py` holds the module (`ValueResidual`) inside a minimal decoder (`TinyLM`). To add MGVR to your
own attention:

```python
class MGVR(nn.Module):
    """Multi-head Gated Value Residual for one attention layer >= 1."""
    def __init__(self, d_model, n_kv_heads, lam=0.5):
        super().__init__()
        G = self.G = n_kv_heads
        self.w_z = nn.Parameter(torch.zeros(G + G * G, d_model))  # gate + mix logits
        self.lam = nn.Parameter(lam * torch.eye(G))               # static mix
        self.scale = d_model ** -0.5

    def forward(self, x, v, v0):
        """x [B,T,d]: normalised attention input; v, v0 [B,T,G,d_h]."""
        G = self.G
        z = F.linear(x.float(), self.w_z) * self.scale
        gate = 2 * torch.sigmoid(z[..., :G])
        mix = self.lam + torch.tanh(z[..., G:]).unflatten(-1, (G, G))
        out = gate[..., None] * v.float() + torch.einsum("btgh,bthd->btgd", mix, v0.float())
        return out.to(v.dtype)

# layer 0:      v0 = v.float()          (pass it to every later layer)
# layer l >= 1: v = mgvr(x, v, v0)      (before attention; the attention kernel is unchanged)
```

Train `w_z` and `lam` with AdamW and **no weight decay**, and keep them in FP32. Weight decay pulls `lam`
towards zero, away from the value residual. Neither parameter draws random numbers at initialisation, so an
MGVR model and a baseline built from the same seed share every other weight.

## Reproduce the synthetic study

```bash
pip install -e .                                   # torch; matplotlib for the plots
python -m mgvr.test_harness                        # 16 checks, CPU, about a minute
python -m mgvr.suite --grid main --out runs/main   # six arms x tasks x learning rates x 3 seeds, one GPU
python -m mgvr.analyze runs/main                   # summary.md, summary.json, plots/
```

The checks cover the formula against a per-(token, head) evaluation, the exact reduction to ResFormer and
to standard attention at initialisation, shared weights across arms, causality, and the task generators.
`--grid cal` runs the calibration grid. The main grid is 162 runs, about nine run-hours, and the suite runs
eight at a time on one GPU.

## Results in this repository

```
results/pretraining/{baseline,mgvr}/eval.jsonl   held-out loss every 50 steps (2.1M tokens, same rows each time)
results/pretraining/throughput.json              training throughput of both arms
results/synthetic/main/                          the main grid: <experiment>/<arm>/lr*/s*/{summary.json,log.jsonl}
results/synthetic/calibration/                   the calibration grid (one seed)
results/synthetic/first_grid/                    the first, miscalibrated grid (settings in each summary.json)
```

The arms in the synthetic study are `none` (Transformer), `resformer`, `resformer_learn` (a learned λ per
layer), `mgvr`, `mgvr_gate` (the gate alone) and `mgvr_mix` (the mix alone).

## Paper

The technical report is [paper.pdf](paper.pdf).

## Citation

```bibtex
@techreport{lehmann2026mgvr,
  title       = {Identity on Demand: Multi-head Gated Value Residuals},
  author      = {Lehmann, Leon},
  institution = {Empero AI Research},
  type        = {Empero AI Technical Report},
  year        = {2026},
  month       = {October},
  url         = {https://github.com/empero-org/mgvr}
}
```

## License

Apache License 2.0, see [LICENSE](LICENSE).
