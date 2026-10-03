"""ONNX Runtime session creation with automatic CUDA preference.

If ``onnxruntime-gpu`` is installed and CUDA init works, sessions run on GPU;
otherwise they transparently fall back to CPU.
"""
import ctypes
import glob
import os
import sys
from typing import List

import onnxruntime as ort

_preloaded = False


def _preload_nvidia_libs() -> None:
    """Preload CUDA/cuDNN shared libs bundled by PyTorch (nvidia-* pip pkgs).

    dlopen resolves the CUDA provider's dependencies (libcublasLt.so.12 etc.)
    against already-loaded libraries, so preloading makes ORT find the libs
    that ship with torch without requiring LD_LIBRARY_PATH.
    """
    global _preloaded
    if _preloaded or sys.platform != "linux":
        return
    _preloaded = True
    for d in glob.glob(os.path.join(os.path.dirname(os.__file__), "site-packages",
                                    "nvidia", "*", "lib")):
        for so in sorted(glob.glob(os.path.join(d, "*.so*"))):
            if os.path.isfile(so):
                try:
                    ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass


def ort_providers(session: ort.InferenceSession) -> List[str]:
    return session.get_providers()


def make_session(model_path: str) -> ort.InferenceSession:
    """Create an InferenceSession, preferring CUDAExecutionProvider."""
    candidates = (["CUDAExecutionProvider", "CPUExecutionProvider"],
                  ["CPUExecutionProvider"])
    last_err: Exception | None = None
    for providers in candidates:
        try:
            if providers[0] == "CUDAExecutionProvider":
                _preload_nvidia_libs()
                opts = ort.SessionOptions()
                # ORT_ENABLE_BASIC skips the FuseConvActivation transformer
                # (FusedConv) whose cudnn v9 graph API needs a newer cuDNN
                # than torch 2.5 bundles (9.1 -> SUBLIBRARY_VERSION_MISMATCH);
                # plain Conv nodes use the cudnn path that works fine.
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                return ort.InferenceSession(model_path, sess_options=opts,
                                            providers=providers)
            return ort.InferenceSession(model_path, providers=providers)
        except Exception as e:  # CUDA provider present but broken -> fall back
            last_err = e
    raise last_err  # pragma: no cover
