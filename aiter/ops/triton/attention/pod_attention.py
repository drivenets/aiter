import logging

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.pod_attention import (
    pod_persistent,
)

logger = logging.getLogger(__name__)


def pod_attention(
    cu_ctr: torch.Tensor,
    # Decode
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    Mp: torch.Tensor,
    Lp: torch.Tensor,
    Op: torch.Tensor,
    locks: torch.Tensor,
    batch_num_block_n: torch.Tensor,
    total_programs: int,
    BLOCK_M: int,
    BLOCK_N: int,
    # causal: bool,
    batch_size: int,
    sm_scale: torch.float16,
    num_warps,
    waves_per_eu,
    # Prefill
    q_pf: torch.Tensor,
    k_pf: torch.Tensor,
    v_pf: torch.Tensor,
    Mp_pf: torch.Tensor,
    Lp_pf: torch.Tensor,
    Op_pf: torch.Tensor,
    locks_pf: torch.Tensor,
    batch_num_block_n_pf: torch.Tensor,
    BLOCK_M_pf: int,
    BLOCK_N_pf: int,
    # causal_pf: bool,
    batch_size_pf: int,
    prefill_ratio: int,
    decode_ratio: int,
):
    """
    POD (Prefill-On-Decode) fused attention for simultaneous prefill and decode execution.
    Launches persistent kernels that execute both operations concurrently on different CUs
    for improved hardware utilization.

    Args:
        cu_ctr (torch.Tensor): CU (Compute Unit) counter for workload distribution.
        q (torch.Tensor): Decode query with shape (batch_size * 1, num_heads, head_dim).
        k (torch.Tensor): Decode key with shape (total_tokens, num_heads, head_dim).
        v (torch.Tensor): Decode value with shape (total_tokens, num_heads, head_dim).
        Mp (torch.Tensor): Decode partial max buffer with shape (total_programs, BLOCK_M).
        Lp (torch.Tensor): Decode partial sum buffer with shape (total_programs, BLOCK_M).
        Op (torch.Tensor): Decode partial output buffer with shape (total_programs, seq_len, head_dim).
        locks (torch.Tensor): Decode synchronization locks.
        batch_num_block_n (torch.Tensor): Decode cumulative BLOCK_N counts per batch.
        total_programs (int): Total number of thread blocks (CTAs) to launch. Should be 2x the
            number of CUs (one for prefill, one for decode per CU).
        BLOCK_M (int): Decode query tile size.
        BLOCK_N (int): Decode key tile size.
        batch_size (int): Decode batch size.
        sm_scale (torch.float16): Softmax scale, typically 1/sqrt(head_dim).
        num_warps (int): Number of warps per CTA.
        waves_per_eu (int): Number of waves per execution unit.
        q_pf (torch.Tensor): Prefill query with shape (batch_size_pf * seq_len_pf, num_heads, head_dim).
        k_pf (torch.Tensor): Prefill key with shape (total_tokens_pf, num_heads, head_dim).
        v_pf (torch.Tensor): Prefill value with shape (total_tokens_pf, num_heads, head_dim).
        Mp_pf (torch.Tensor): Prefill partial max buffer.
        Lp_pf (torch.Tensor): Prefill partial sum buffer.
        Op_pf (torch.Tensor): Prefill partial output buffer.
        locks_pf (torch.Tensor): Prefill synchronization locks.
        batch_num_block_n_pf (torch.Tensor): Prefill cumulative BLOCK_N counts per batch.
        BLOCK_M_pf (int): Prefill query tile size.
        BLOCK_N_pf (int): Prefill key tile size.
        batch_size_pf (int): Prefill batch size.
        prefill_ratio (int): Ratio of workload assigned to prefill workgroups.
        decode_ratio (int): Ratio of workload assigned to decode workgroups.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (decode_output, prefill_output) with shapes
            matching respective query tensors.
    """
    # shape constraints
    HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = q.shape[-1], k.shape[-1], v.shape[-1]
    assert (
        HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    ), "Incompatible Q/K/V Hidden Dimensions"
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    # Calculate Decode Params
    N_CTX_Q = q.shape[0] // batch_size
    N_CTX_K = k.shape[0]  # This is the sum of all ctx_n in a batch
    H = q.shape[1]
    H_K = k.shape[1]  # GQA: may differ from H

    # Note: do NOT pre-multiply by log2(e) — the lean_atten kernel does this internally
    qk_scale = sm_scale

    # We assume the kernel functions fused by pod attention are persistent kernel functions
    # For gfx942, we launch total 608 WGs. Each CU will get 2 WG --- one WG will be doing decode and one prefill
    # For different decode:prefill ratios, assign (decode+prefill)*304 number of WGs
    total_wgs = total_programs // 2

    (
        num_m_blocks,
        num_n_blocks,
        high_load_wgs,
        max_tiles_per_wg,
        tiles_per_head,
        effective_programs,
        num_splits,
        even_split,
    ) = get_num_splits_and_buffer_sizes(
        False,  # causal
        batch_size,
        N_CTX_Q,
        N_CTX_K,
        H,
        BLOCK_M,
        BLOCK_N,
        total_wgs,
    )

    # The lean kernel reads/writes BLOCK_M rows per head regardless of actual N_CTX_Q.
    # Pad Q and output to BLOCK_M-aligned rows to avoid OOB reads/writes.
    padded_rows = max(q.shape[0], BLOCK_M * batch_size)
    if q.shape[0] < padded_rows:
        q_padded = torch.zeros(padded_rows, H, HEAD_DIM_K, dtype=q.dtype, device=q.device)
        q_padded[:q.shape[0]] = q
        q = q_padded
    o_padded = torch.empty(padded_rows, H, HEAD_DIM_K, dtype=v.dtype, device=q.device)
    o = o_padded[:N_CTX_Q * batch_size]

    # Calculate Prefill Params
    N_CTX_Q_pf = q_pf.shape[0] // batch_size_pf
    N_CTX_K_pf = k_pf.shape[0]  # This is the sum of all ctx_n in a batch
    H_K_pf = k_pf.shape[1]  # GQA: may differ from H

    # MASKED_BLOCKS is used for prefill/causal for BLOCK_M > BLOCK_N
    # For gfx942, BLOCK_M=128, BLOCK_N=64 is better for performance
    MASKED_BLOCKS = BLOCK_M_pf // BLOCK_N_pf

    # Only support BLOCK_M is multiple of BLOCK_N
    assert BLOCK_M_pf % BLOCK_N_pf == 0

    (
        num_m_blocks_pf,
        num_n_blocks_pf,
        high_load_wgs_pf,
        max_tiles_per_wg_pf,
        tiles_per_head_pf,
        effective_programs_pf,
        num_splits_pf,
        even_split_pf,
    ) = get_num_splits_and_buffer_sizes(
        True,  # causal
        batch_size_pf,
        N_CTX_Q_pf,
        N_CTX_K_pf,
        H,
        BLOCK_M_pf,
        BLOCK_N_pf,
        total_wgs,
    )
    logger.debug(
        "POD prefill LA: num_m=%d high_load=%d max_tiles=%d tiles_per_head=%d "
        "total_wgs=%d BLOCK_M=%d BLOCK_N=%d MASKED=%d bs=%d",
        num_m_blocks_pf, high_load_wgs_pf, max_tiles_per_wg_pf,
        tiles_per_head_pf, total_wgs, BLOCK_M_pf, BLOCK_N_pf,
        MASKED_BLOCKS, batch_size_pf,
    )
    logger.debug("POD launching %d kernels", total_programs)

    grid = (total_programs, 1, 1)

    # Same padding for prefill Q and output
    padded_rows_pf = max(q_pf.shape[0], BLOCK_M_pf * batch_size_pf)
    if q_pf.shape[0] < padded_rows_pf:
        q_pf_padded = torch.zeros(padded_rows_pf, H, HEAD_DIM_K, dtype=q_pf.dtype, device=q_pf.device)
        q_pf_padded[:q_pf.shape[0]] = q_pf
        q_pf = q_pf_padded
    o_pf_padded = torch.empty(padded_rows_pf, H, HEAD_DIM_K, dtype=v_pf.dtype, device=q_pf.device)
    o_pf = o_pf_padded[:N_CTX_Q_pf * batch_size_pf]

    # TODO: need to tune
    max_output_tile_cnt = 16

    # GQA group size for kernel-level head mapping
    gqa_group_size = H // H_K
    HEAD_DIM_PADDED = triton.next_power_of_2(HEAD_DIM_K)
    if HEAD_DIM_PADDED < 16:
        HEAD_DIM_PADDED = 16
    MASKED_BLOCKS_DEC = BLOCK_M // BLOCK_N

    pod_kernel = pod_persistent[grid](
        cu_ctr,
        # Decode positional arguments
        q, k, v,
        Mp, Lp, Op, o,
        batch_num_block_n, locks,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        N_CTX_Q,  # n_ctx_q_rows for decode
        Op.stride(0), Op.stride(1), Op.stride(2),
        qk_scale,  # sm_scale (log2-scaled)
        # Prefill positional arguments
        q_pf, k_pf, v_pf,
        Mp_pf, Lp_pf, Op_pf, o_pf,
        batch_num_block_n_pf, locks_pf,
        q_pf.stride(0), q_pf.stride(1), q_pf.stride(2),
        k_pf.stride(0), k_pf.stride(1), k_pf.stride(2),
        v_pf.stride(0), v_pf.stride(1), v_pf.stride(2),
        o_pf.stride(0), o_pf.stride(1), o_pf.stride(2),
        N_CTX_Q_pf,  # n_ctx_q_rows for prefill
        Op_pf.stride(0), Op_pf.stride(1), Op_pf.stride(2),
        # Constexpr — keyword arguments
        HEAD_DIM=HEAD_DIM_PADDED,
        HEAD_DIM_ORIG=HEAD_DIM_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        MASKED_BLOCKS_DEC=MASKED_BLOCKS_DEC,
        batch_size=batch_size,
        num_m_blocks=num_m_blocks,
        num_n_blocks=num_n_blocks,
        high_load_wgs=high_load_wgs,
        max_tiles_per_wg=max_tiles_per_wg,
        tiles_per_head=tiles_per_head,
        num_splits=num_splits,
        BLOCK_M_pf=BLOCK_M_pf,
        BLOCK_N_pf=BLOCK_N_pf,
        MASKED_BLOCKS=MASKED_BLOCKS,
        batch_size_pf=batch_size_pf,
        num_m_blocks_pf=num_m_blocks_pf,
        num_n_blocks_pf=num_n_blocks_pf,
        high_load_wgs_pf=high_load_wgs_pf,
        max_tiles_per_wg_pf=max_tiles_per_wg_pf,
        tiles_per_head_pf=tiles_per_head_pf,
        num_splits_pf=num_splits_pf,
        prefill_ratio=prefill_ratio,
        decode_ratio=decode_ratio,
        max_output_tile_cnt=max_output_tile_cnt,
        gqa_group_size=gqa_group_size,
        total_programs_half=total_wgs,
        waves_per_eu=waves_per_eu,
        num_warps=num_warps,
    )
    logger.debug(
        "POD kernel: %d registers, %d spills",
        pod_kernel.n_regs, pod_kernel.n_spills,
    )

    return o, o_pf


def get_num_splits_and_buffer_sizes(
    causal,
    batch_size,
    max_seqlen_q,
    max_seqlen_k,
    num_heads,
    BLOCK_M,
    BLOCK_N,
    num_SMs,
):
    """
    Calculates workload distribution parameters for POD attention stream-K scheduling.
    Matches lean attention's scheduling logic — schedules over Q heads (not K heads),
    with GQA head mapping handled inside the kernel via gqa_group_size.

    Args:
        causal (bool): Causal masking mode.
        batch_size (int): Batch size.
        max_seqlen_q (int): Maximum query sequence length.
        max_seqlen_k (int): Maximum key sequence length.
        num_heads (int): Number of Q heads (NOT K heads).
        BLOCK_M (int): Query tile size.
        BLOCK_N (int): Key tile size.
        num_SMs (int): Number of streaming multiprocessors (CTAs available).

    Returns:
        Tuple: (num_m_blocks, num_n_blocks, high_load_tbs, max_tiles_per_tb,
            tiles_per_head, total_programs, num_splits, even_split).
    """
    num_m_blocks = (max_seqlen_q + BLOCK_M - 1) // BLOCK_M
    num_n_blocks = (max_seqlen_k + BLOCK_N - 1) // BLOCK_N

    if max_seqlen_q == 1:
        causal = False

    tiles_per_head = 0
    if causal:
        for i in range(0, num_m_blocks):
            tiles_per_head += (((i + 1) * BLOCK_M) + BLOCK_N - 1) // BLOCK_N
        tiles_per_head = tiles_per_head * batch_size
    else:
        tiles_per_head = num_m_blocks * num_n_blocks

    # Schedule over ALL Q heads (GQA mapping done in kernel)
    total_tiles = tiles_per_head * num_heads

    # POD uses a fixed grid size — no grid reduction.
    # Idle CTAs (when total_tiles < num_SMs) safely get 0 tiles
    # via the max_tiles_per_tb=1 special case.
    lean_griddimz = num_SMs

    max_tiles_per_tb = (total_tiles + lean_griddimz - 1) // lean_griddimz

    num_splits = 0
    even_split = False
    if max_tiles_per_tb <= 1:
        # Many CTAs but few tiles — each CTA processes at most 1 tile.
        # An output tile may span multiple N-blocks (up to num_n_blocks),
        # so that many CTAs contribute. Use the even_split formula.
        max_tiles_per_tb = 1
        even_split = True
        high_load_tbs = total_tiles
        num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 2) // max_tiles_per_tb)
    elif total_tiles % lean_griddimz == 0:
        even_split = True
        num_splits = 1 + ((num_n_blocks + max_tiles_per_tb - 2) // (max_tiles_per_tb))
        high_load_tbs = total_tiles - ((max_tiles_per_tb - 1) * lean_griddimz)
    else:
        even_split = False
        num_splits = 1 + (
            (num_n_blocks + max_tiles_per_tb - 3) // (max_tiles_per_tb - 1)
        )
        high_load_tbs = total_tiles - ((max_tiles_per_tb - 1) * lean_griddimz)

    num_n_blocks = num_n_blocks // batch_size

    return (
        num_m_blocks,
        num_n_blocks,
        high_load_tbs,
        max_tiles_per_tb,
        tiles_per_head,
        lean_griddimz,
        num_splits,
        even_split,
    )
