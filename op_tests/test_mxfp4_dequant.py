# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tests for the native MXFP4 dequant op ``dynamic_per_group_scaled_dequant``.

The op is the reverse of ``dynamic_per_group_scaled_quant``: it decodes packed
e2m1 fp4 ``[N, D/2]`` + e8m0 group scales ``[N, D/group_size]`` into bf16/fp32
``[N, D]`` using the gfx950 hardware fp4->f32 intrinsic.

Two things are pinned here:
  * round-trip through the real encoder is BIT-EXACT vs the pure-torch reference
    (``fp4_utils.mxfp4_to_f32`` x ``e8m0_to_f32``) -- 0 element mismatches;
  * the documented subnormal-flush: for hand-crafted raw e8m0 scale bytes 0/1
    (scale <= 2^-126, which the encoder never emits) the hardware intrinsic
    flushes subnormal results (|x| < 2^-126) to +/-0.0 while the reference keeps
    them. This test asserts that difference so the behavior is a pinned contract,
    not a surprise.
"""

import pytest
import torch

import aiter
from aiter.utility import fp4_utils as fu

torch.set_default_device("cuda")

F32_MIN_NORMAL = 2.0 ** (-126)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a ROCm GPU"
)


def _ref_dequant_f32(packed, scales, N, D, group_size):
    """Pure-torch reference: mxfp4_to_f32(packed) * e8m0_to_f32(scale)."""
    g = D // group_size
    vals = fu.mxfp4_to_f32(packed).view(N, g, group_size)
    scale = fu.e8m0_to_f32(scales).view(N, g, 1)
    return (vals * scale).view(N, D).float()


def _safe_scale_bytes(N, ng, seed):
    """Random e8m0 scale bytes that never produce subnormal outputs.

    Bytes 0/1 give scale <= 2^-126 (the FTZ region tested separately) and 0xFF
    is NaN; restrict to [2, 0xFE] so every decoded value stays normal and the
    kernel must be bit-exact vs the reference.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(
        2, 0xFF, (N, ng), generator=g, dtype=torch.int32, device="cuda"
    ).to(torch.uint8)


@pytest.mark.parametrize("N", [1, 3, 128, 4096])
def test_roundtrip_encoder_bitexact(N):
    """fp32 kernel output == pure-torch reference EXACTLY on ENCODER outputs.

    ``per_1x32_f4_quant_hip`` is group_size=32 only, so this is the real
    encode->decode round trip at group_size=32.
    """
    torch.manual_seed(0)
    group_size, D = 32, 128
    x = torch.randn((N, D), dtype=torch.bfloat16, device="cuda")

    packed, scales = aiter.per_1x32_f4_quant_hip(x)  # encoder (shuffle=False)

    ref = _ref_dequant_f32(packed, scales, N, D, group_size)

    out = torch.empty((N, D), dtype=torch.float32, device="cuda")
    aiter.dynamic_per_group_scaled_dequant(
        out, packed, scales, group_size=group_size, shuffle_scale=False
    )

    n_mism = int(((out != ref) & ~(out.isnan() & ref.isnan())).sum().item())
    assert n_mism == 0, (
        f"N={N} D={D} gs={group_size}: {n_mism} mismatches vs reference "
        f"(expected 0 for encoder-produced inputs)"
    )
    torch.testing.assert_close(out, ref, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("N", [1, 3, 128])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_dequant_bitexact_synth_scales(N, group_size):
    """fp32 kernel output == reference EXACTLY for group_size in {32,64,128}.

    The encoder is 32-only, so to exercise group_size 64/128 we synthesize
    inputs directly: random packed fp4 bytes + random (non-subnormal) e8m0
    scale bytes, D a multiple of group_size.
    """
    D = group_size * 4  # D % group_size == 0
    ng = D // group_size
    g = torch.Generator(device="cuda").manual_seed(100 + group_size + N)
    packed = torch.randint(
        0, 256, (N, D // 2), generator=g, dtype=torch.int32, device="cuda"
    ).to(torch.uint8)
    scales = _safe_scale_bytes(N, ng, seed=200 + group_size + N)

    ref = _ref_dequant_f32(packed, scales, N, D, group_size)

    out = torch.empty((N, D), dtype=torch.float32, device="cuda")
    aiter.dynamic_per_group_scaled_dequant(
        out, packed, scales, group_size=group_size, shuffle_scale=False
    )
    n_mism = int(((out != ref) & ~(out.isnan() & ref.isnan())).sum().item())
    assert n_mism == 0, (
        f"N={N} D={D} gs={group_size}: {n_mism} mismatches vs reference"
    )


@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_roundtrip_bf16_out(group_size):
    """bf16 kernel output == bf16(reference) exactly (synth non-subnormal input)."""
    N, D = 64, group_size * 2
    ng = D // group_size
    g = torch.Generator(device="cuda").manual_seed(7 + group_size)
    packed = torch.randint(
        0, 256, (N, D // 2), generator=g, dtype=torch.int32, device="cuda"
    ).to(torch.uint8)
    scales = _safe_scale_bytes(N, ng, seed=17 + group_size)

    ref_bf16 = _ref_dequant_f32(packed, scales, N, D, group_size).to(torch.bfloat16)

    out = torch.empty((N, D), dtype=torch.bfloat16, device="cuda")
    aiter.dynamic_per_group_scaled_dequant(
        out, packed, scales, group_size=group_size, shuffle_scale=False
    )
    assert torch.equal(out, ref_bf16), f"bf16 out mismatch gs={group_size}"


@pytest.mark.parametrize("scale_byte", [0, 1])
def test_subnormal_flush_is_pinned(scale_byte):
    """DOCUMENTED divergence: raw e8m0 scale bytes 0/1 => subnormal f32 result.

    The gfx950 intrinsic flushes it to +/-0.0 (FTZ) while the pure-torch
    reference preserves the subnormal. This is unreachable through the real
    encoder (it never emits scale bytes 0/1), so we craft the packed/scale
    tensors by hand. We assert BOTH that the kernel returns exactly 0 AND that
    the reference is a nonzero subnormal, pinning the expected difference.
    """
    N, D, group_size = 1, 32, 32
    # e8m0 byte 0 -> f32 2^-126 (reference special case); byte 1 -> 2^-126.
    # fp4 nibble 2 (value 1.0) and 1 (value 0.5): 1.0 * 2^-126 == 2^-126 (the
    # boundary, not subnormal); 0.5 * 2^-126 == 2^-127 -> subnormal -> flushed.
    # Pack every nibble as 0.5 (nibble 1) so both nibbles per byte decode to 0.5.
    nibble = 0x1  # value 0.5
    packed = torch.full(
        (N, D // 2), (nibble << 4) | nibble, dtype=torch.uint8, device="cuda"
    )
    scales = torch.full(
        (N, D // group_size), scale_byte, dtype=torch.uint8, device="cuda"
    )

    ref = _ref_dequant_f32(packed, scales, N, D, group_size)

    out = torch.empty((N, D), dtype=torch.float32, device="cuda")
    aiter.dynamic_per_group_scaled_dequant(
        out, packed, scales, group_size=group_size, shuffle_scale=False
    )

    # Reference keeps the subnormal (0.5 * 2^-126 = 2^-127 ~= 5.88e-39).
    assert torch.all(ref.abs() > 0), (
        f"expected nonzero subnormal reference for scale_byte={scale_byte}, "
        f"got {ref.flatten()[:4].tolist()}"
    )
    assert torch.all(ref.abs() < F32_MIN_NORMAL), (
        "reference values should be subnormal (|x| < 2^-126)"
    )
    # Kernel flushes them to +/-0.0 (documented hardware FTZ behavior).
    assert torch.all(out == 0), (
        f"expected kernel to flush subnormals to 0 for scale_byte={scale_byte}, "
        f"got {out.flatten()[:4].tolist()}"
    )
    # And confirm this is the ONLY difference: they disagree exactly on the
    # flushed subnormals.
    assert not torch.equal(out, ref), (
        "kernel and reference should differ on subnormal (flush vs keep)"
    )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
