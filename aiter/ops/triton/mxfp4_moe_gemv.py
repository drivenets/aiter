# SPDX-License-Identifier: MIT
# Triton MXFP4 MoE GEMV kernel for decode (small M).
# Processes unshuffled, minimally-padded weights to avoid CK-tile 256-alignment waste.

import torch
import triton
import triton.language as tl


@triton.jit
def _e8m0_to_f32(scale_u8):
    """Convert E8M0 scale byte to float32: value = 2^(e - 127)."""
    return tl.exp2((scale_u8.to(tl.float32) - 127.0))


@triton.jit
def _mxfp4_moe_gemv_kernel(
    # Inputs
    x_ptr,              # [flat_tokens, K] bf16
    w_ptr,              # [E, N, K_packed] uint8 (fp4x2, unshuffled, row-major)
    scale_ptr,          # [E, N, K_groups] uint8 (e8m0, unshuffled)
    # MoE routing
    sorted_ids_ptr,         # [num_sorted] int32
    sorted_expert_ids_ptr,  # [num_blocks] int32
    num_valid_ids_ptr,      # [2] int32 — [0]=num_valid, [1]=num_tokens
    sorted_weights_ptr,     # [num_sorted] float32
    # Output
    out_ptr,            # [out_tokens, N_out] float32 (accumulate in fp32!)
    # Dims
    K_packed: tl.constexpr,
    K_groups: tl.constexpr,
    N: tl.constexpr,
    N_out: tl.constexpr,
    max_token_id,       # for bounds checking
    stride_x_t,
    stride_w_e,
    stride_w_n,
    stride_s_e,
    stride_s_n,
    block_m: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,  # 32
    IS_GEMM2: tl.constexpr,
):
    """
    Grid: (num_sorted_tokens, ceil(N / BLOCK_N))
    Each program computes BLOCK_N output values for one sorted token.
    """
    pid_token = tl.program_id(0)
    pid_n = tl.program_id(1)

    num_valid = tl.load(num_valid_ids_ptr)
    if pid_token >= num_valid:
        return

    # Decode sorted_id
    packed_id = tl.load(sorted_ids_ptr + pid_token)
    token_id = packed_id & 0xFFFFFF
    topk_id = packed_id >> 24

    # Skip padding entries (token_id >= max_token_id)
    if token_id >= max_token_id:
        return

    # Get expert ID
    block_idx = pid_token // block_m
    expert_id = tl.load(sorted_expert_ids_ptr + block_idx)

    # N range for this tile
    n_base = pid_n * BLOCK_N
    n_offsets = n_base + tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N

    # Accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)

    stride_x_t = tl.cast(stride_x_t, tl.int64)
    stride_w_e = tl.cast(stride_w_e, tl.int64)
    stride_w_n = tl.cast(stride_w_n, tl.int64)
    stride_s_e = tl.cast(stride_s_e, tl.int64)
    stride_s_n = tl.cast(stride_s_n, tl.int64)

    w_base = w_ptr + expert_id * stride_w_e
    s_base = scale_ptr + expert_id * stride_s_e

    # Iterate over K in groups of GROUP_SIZE (32 elements = 16 packed bytes)
    for g in range(K_groups):
        # Load scale for each N in tile at group g
        scale_vals = tl.load(
            s_base + n_offsets * stride_s_n + g,
            mask=n_mask, other=127
        ).to(tl.uint8)
        scale_f32 = _e8m0_to_f32(scale_vals)  # [BLOCK_N]

        # Load fp4x2 weights: [BLOCK_N, GROUP_SIZE//2] bytes
        w_packed_base = g * (GROUP_SIZE // 2)
        w_packed_offsets = w_packed_base + tl.arange(0, GROUP_SIZE // 2)
        w_bytes = tl.load(
            w_base + n_offsets[:, None] * stride_w_n + w_packed_offsets[None, :],
            mask=n_mask[:, None], other=0
        )  # [BLOCK_N, 16] uint8

        # Unpack fp4x2
        lo = (w_bytes & 0xF)
        hi = (w_bytes >> 4) & 0xF

        # Dequant FP4 E2M1 - lo nibble
        lo_sign = ((lo >> 3) & 1).to(tl.float32)
        lo_exp = ((lo >> 1) & 3).to(tl.int32)
        lo_mant = (lo & 1).to(tl.float32)
        lo_is_zero = (lo_exp == 0) & (lo_mant == 0.0)
        lo_is_sub = (lo_exp == 0) & (lo_mant != 0.0)
        lo_abs = tl.where(lo_is_zero, 0.0,
                 tl.where(lo_is_sub, 0.5,
                 tl.exp2((lo_exp - 1).to(tl.float32)) * (1.0 + 0.5 * lo_mant)))
        lo_val = tl.where(lo_sign != 0.0, -lo_abs, lo_abs)

        # Dequant FP4 E2M1 - hi nibble
        hi_sign = ((hi >> 3) & 1).to(tl.float32)
        hi_exp = ((hi >> 1) & 3).to(tl.int32)
        hi_mant = (hi & 1).to(tl.float32)
        hi_is_zero = (hi_exp == 0) & (hi_mant == 0.0)
        hi_is_sub = (hi_exp == 0) & (hi_mant != 0.0)
        hi_abs = tl.where(hi_is_zero, 0.0,
                 tl.where(hi_is_sub, 0.5,
                 tl.exp2((hi_exp - 1).to(tl.float32)) * (1.0 + 0.5 * hi_mant)))
        hi_val = tl.where(hi_sign != 0.0, -hi_abs, hi_abs)

        # Apply scale
        lo_scaled = lo_val * scale_f32[:, None]  # [BLOCK_N, 16]
        hi_scaled = hi_val * scale_f32[:, None]

        # Load activation: even and odd elements
        x_base_k = g * GROUP_SIZE
        x_even = tl.load(
            x_ptr + token_id * stride_x_t + x_base_k + tl.arange(0, GROUP_SIZE // 2) * 2,
            mask=(x_base_k + tl.arange(0, GROUP_SIZE // 2) * 2) < (K_groups * GROUP_SIZE),
            other=0.0
        ).to(tl.float32)  # [16]

        x_odd = tl.load(
            x_ptr + token_id * stride_x_t + x_base_k + tl.arange(0, GROUP_SIZE // 2) * 2 + 1,
            mask=(x_base_k + tl.arange(0, GROUP_SIZE // 2) * 2 + 1) < (K_groups * GROUP_SIZE),
            other=0.0
        ).to(tl.float32)  # [16]

        # Dot product
        acc += tl.sum(lo_scaled * x_even[None, :], axis=1)
        acc += tl.sum(hi_scaled * x_odd[None, :], axis=1)

    # Apply routing weight for GEMM2
    if IS_GEMM2:
        routing_weight = tl.load(sorted_weights_ptr + pid_token)
        acc = acc * routing_weight

    # Atomic add to float32 output
    out_offsets = token_id * N_out + n_offsets
    tl.atomic_add(out_ptr + out_offsets, acc, mask=n_mask)


def mxfp4_moe_gemv(
    x: torch.Tensor,              # [flat_tokens, K] bf16
    weight: torch.Tensor,         # [E, N, K_packed] uint8
    scale: torch.Tensor,          # [E, N, K_groups] uint8
    sorted_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,  # [2] int32
    sorted_weights: torch.Tensor,
    output: torch.Tensor,         # [out_tokens, N_out] float32
    topk: int,
    block_m: int,
    is_gemm2: bool = False,
    max_token_id: int = None,
):
    """
    Triton MXFP4 MoE GEMV for decode.
    weight/scale are unshuffled, minimally-padded (K aligned to 32 only).
    Output is float32 for correct atomic accumulation.
    """
    E, N, K_packed = weight.shape
    K_groups = scale.shape[2]
    N_out = output.shape[1]

    if max_token_id is None:
        max_token_id = x.shape[0]

    BLOCK_N = min(64, N)
    GROUP_SIZE = 32

    num_sorted = sorted_ids.shape[0]
    grid = (num_sorted, triton.cdiv(N, BLOCK_N))

    # Zero output for atomic accumulation
    output.zero_()

    _mxfp4_moe_gemv_kernel[grid](
        x, weight.view(torch.uint8), scale.view(torch.uint8),
        sorted_ids, sorted_expert_ids, num_valid_ids,
        sorted_weights if sorted_weights is not None else x,
        output,
        K_packed=K_packed, K_groups=K_groups,
        N=N, N_out=N_out,
        max_token_id=max_token_id,
        stride_x_t=x.stride(0),
        stride_w_e=weight.stride(0),
        stride_w_n=weight.stride(1),
        stride_s_e=scale.stride(0),
        stride_s_n=scale.stride(1),
        block_m=block_m,
        BLOCK_N=BLOCK_N,
        GROUP_SIZE=GROUP_SIZE,
        IS_GEMM2=is_gemm2,
    )

    return output
