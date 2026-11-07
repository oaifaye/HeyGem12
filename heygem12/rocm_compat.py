"""ROCm (AMD) workarounds.

On some ROCm/Windows builds MIOpen's BatchNorm kernels fail to compile at
runtime (hiprtc: "'type_traits' file not found"), raising
``miopenStatusUnknownError`` from ``torch.batch_norm``. For inference we only
need eval-mode BN, which is a simple affine op, so we compute it with plain
tensor math instead. Convolutions keep using MIOpen.

Control with env var HEYGEM12_BN_FIX = auto (default; only on ROCm GPU) | 1 | 0.
"""
import os
import types

import torch
from torch import nn


def _plain_bn_forward(self, x):
    if self.training or self.running_mean is None:
        return self._heygem_orig_forward(x)
    shape = [1, -1] + [1] * (x.dim() - 2)
    inv = torch.rsqrt(self.running_var + self.eps)
    scale = inv if self.weight is None else self.weight * inv
    shift = -self.running_mean * scale
    if self.bias is not None:
        shift = shift + self.bias
    return x * scale.view(shape).to(x.dtype) + shift.view(shape).to(x.dtype)


def _wanted(device) -> bool:
    mode = os.environ.get("HEYGEM12_BN_FIX", "auto").lower()
    if mode in ("0", "off", "false"):
        return False
    if mode in ("1", "on", "true"):
        return True
    return bool(getattr(torch.version, "hip", None)) and str(device).startswith("cuda")


def patch_batchnorm(model: nn.Module, device) -> int:
    """Replace eval-mode BatchNorm forward with plain math on ROCm GPUs."""
    if not _wanted(device):
        return 0
    n = 0
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not hasattr(m, "_heygem_orig_forward"):
            m._heygem_orig_forward = m.forward
            m.forward = types.MethodType(_plain_bn_forward, m)
            n += 1
    return n
