"""
Fused FP8 blockscale GEMV for M=1 decode.
Replaces splitK GEMM + reduce with a single kernel.
Uses gfx950 hardware v_cvt_f32_fp8 for single-cycle FP8→F32.
"""

import os
import torch
from torch.utils.cpp_extension import load as _load_ext

_gemv_module = None
_KERNEL_SRC = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "csrc", "kernels", "fp8_gemv_blockscale.cu",
)


def _get_module():
    global _gemv_module
    if _gemv_module is None:
        _gemv_module = _load_ext(
            name="fp8_gemv_blockscale_ext",
            sources=[_KERNEL_SRC],
            extra_include_paths=["/opt/rocm-7.2.0/include"],
            extra_cflags=["-O3"],
            extra_cuda_cflags=[
                "-O3",
                "--offload-arch=gfx950",
                "-DUSE_ROCM=1",
                "-U__HIP_NO_HALF_CONVERSIONS__",
                "-U__HIP_NO_HALF_OPERATORS__",
                "-DHIP_ENABLE_GFX950_OCP_BUILTINS=1",
            ],
            verbose=False,
        )
    return _gemv_module


def fp8_gemv_blockscale(
    x: torch.Tensor,       # [M, K] or [K] FP8 (M must be 1)
    w: torch.Tensor,       # [N, K] FP8 row-major (original weight)
    x_scale: torch.Tensor, # [M, K/128] or [K/128] FP32
    w_scale: torch.Tensor, # [N/128, K/128] FP32 (original layout)
    y: torch.Tensor = None,  # [M, N] or [N] BF16 (pre-allocated optional)
) -> torch.Tensor:
    """
    Fused FP8 blockscale GEMV: Y = X @ W^T with block-wise scales.
    Specialized for M=1. No splitK reduce needed.
    """
    mod = _get_module()

    x_flat = x.view(-1)
    xs_flat = x_scale.view(-1)
    N = w.size(0)

    if y is None:
        y = torch.empty(N, dtype=torch.bfloat16, device=x.device)
    y_flat = y.view(-1)

    mod.fp8_gemv_blockscale(x_flat, w, xs_flat, w_scale, y_flat)
    return y.view(1, N) if x.dim() == 2 else y
