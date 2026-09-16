"""Leaky integrate-and-fire network. Superset of app.py + server.py versions."""
import numpy as np
import torch
import torch.nn as nn


class FlyBrainLIF(nn.Module):
    """Sparse LIF: V = V*decay + W*S + ext (+ noise); fire @ v_thr, refractory.

    The offline script used module-level constants; the server needs
    runtime-tunable dt/tau/v_thr + reset(). This class supports both:
    ``FlyBrainLIF(W, n)`` behaves like the old app.py version, while
    ``update_params`` / ``reset`` / ``step(ext, noise=..)`` match server.py.
    """

    def __init__(self, W, n: int, dt: float = 1.0, tau_mem: float = 10.0,
                 v_thr: float = 1.0, v_reset: float = 0.0,
                 refractory: int = 2, device: str | None = None):
        super().__init__()
        self.W = W
        self.n = n
        self.dt = dt
        self.tau_mem = tau_mem
        self.v_thr = v_thr
        self.v_reset = v_reset
        self.refractory = refractory
        self.decay = float(np.exp(-dt / tau_mem))
        dev = device or W.device
        self.register_buffer("V", torch.zeros(n, device=dev))
        self.register_buffer("S", torch.zeros(n, device=dev))
        self.register_buffer(
            "ref", torch.zeros(n, device=dev, dtype=torch.long))

    def update_params(self, dt=None, tau_mem=None, v_thr=None, noise=None):
        if dt is not None:
            self.dt = dt
        if tau_mem is not None:
            self.tau_mem = tau_mem
            self.decay = float(np.exp(-self.dt / self.tau_mem))
        if v_thr is not None:
            self.v_thr = v_thr

    @torch.no_grad()
    def step(self, ext, noise: float = 0.0):
        I = torch.sparse.mm(self.W, self.S.unsqueeze(1)).squeeze(1)
        self.V = self.V * self.decay + I + ext
        if noise > 0:
            self.V += torch.randn(self.n, device=self.V.device) * noise
        fired = (self.V >= self.v_thr) & (self.ref == 0)
        self.V = torch.where(
            fired,
            torch.as_tensor(self.v_reset, device=self.V.device, dtype=self.V.dtype),
            self.V,
        )
        self.ref = torch.where(
            fired, self.refractory, self.ref - 1).clamp_(min=0)
        self.S = fired.float()
        return self.S

    def reset(self):
        self.V.zero_()
        self.S.zero_()
        self.ref.zero_()
