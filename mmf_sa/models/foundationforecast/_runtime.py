"""Process-level setup required before foundationforecast models are built."""
import importlib
import logging
import os

from mmf_sa.exceptions import ModelInitializationError

_logger = logging.getLogger(__name__)

_TIREX2_SLSTM_MODULE = "tirex2.model.component.flashrnn_slstm"
_TIREX2_BACKEND_FN = "_flashrnn_backend"

_runtime_ready = False


def ensure_runtime() -> None:
    """Prepare the current process for foundationforecast. Safe to call repeatedly."""
    global _runtime_ready
    if _runtime_ready:
        return
    # Must be set before JAX is imported; foundationforecast imports every backend eagerly.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import mlflow
    mlflow.transformers.autolog(disable=True)
    patch_tirex2_slstm_backend()
    _runtime_ready = True


def patch_tirex2_slstm_backend() -> None:
    """Route TiRex-2's sLSTM layers to flashrnn's pure-PyTorch backend on CUDA.

    The fused "cuda" backend JIT-compiles a CUDA extension with nvcc, which fails on
    DBR 18 ML (CUDA 12.9 toolkit without cuSOLVER headers vs. torch built for CUDA 13).
    The mLSTM layers keep their Triton kernels.
    """
    try:
        module = importlib.import_module(_TIREX2_SLSTM_MODULE)
    except ImportError:
        _logger.debug("tirex2 is not installed; skipping the TiRex-2 sLSTM backend patch.")
        return
    original = getattr(module, _TIREX2_BACKEND_FN, None)
    if original is None:
        raise ModelInitializationError(
            f"{_TIREX2_SLSTM_MODULE}.{_TIREX2_BACKEND_FN} no longer exists, so MMF cannot "
            "select TiRex-2's pure-PyTorch sLSTM backend. Install the timecopilot-tirex2 "
            "version pinned in requirements-foundationforecast.txt."
        )
    if getattr(original, "_mmf_patched", False):
        return

    def _flashrnn_backend(device):
        return "vanilla" if device == "cuda" else original(device)

    _flashrnn_backend._mmf_patched = True
    setattr(module, _TIREX2_BACKEND_FN, _flashrnn_backend)
