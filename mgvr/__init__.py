"""MGVR: Multi-head Gated Value Residuals -- a minimal reference implementation and the
synthetic study of the paper "Identity on Demand: Multi-head Gated Value Residuals"."""
from .model import VRES, Config, TinyLM, ValueResidual

__all__ = ["VRES", "Config", "TinyLM", "ValueResidual"]
