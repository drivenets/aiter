import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

import importlib.util
from pathlib import Path

file_path = (Path(__file__).parent / "lean_atten.py").resolve()
module_name = "la_persistent"
spec = importlib.util.spec_from_file_location(module_name, file_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@triton.jit
def get_cu_id():
    # HW_ID Register bit structure for CDNA
    #   CU_ID       11:8    Compute Unit the wave is assigned to.
    #   SE_ID       14:13   Shader Engine (gfx942)
    # XCC_ID Register bit structure for 942/950
    #   XCC_ID      3:0     XCC the wave is assigned to.
    cu_id, se_id, xcc_id = tl.inline_asm_elementwise(
        asm="""
        s_getreg_b32 $0, hwreg(HW_REG_HW_ID, 8, 4)
        s_getreg_b32 $1, hwreg(HW_REG_HW_ID, 13, 2)
        s_getreg_b32 $2, hwreg(HW_REG_XCC_ID, 0, 4)
        s_waitcnt lgkmcnt(0)
        """,
        constraints=("=s,=s,=s"),
        args=[],
        dtype=(tl.int32, tl.int32, tl.int32),
        is_pure=False,
        pack=1,
    )
    return (cu_id, se_id, xcc_id)


_pod_persistent_repr = make_kernel_repr(
    "pod_persistent",
    [
        "HEAD_DIM",
        "HEAD_DIM_ORIG",
        "BLOCK_M",
        "BLOCK_N",
        "MASKED_BLOCKS_DEC",
        "batch_size",
        "num_m_blocks",
        "num_n_blocks",
        "high_load_wgs",
        "max_tiles_per_wg",
        "tiles_per_head",
        "num_splits",
        "BLOCK_M_pf",
        "BLOCK_N_pf",
        "MASKED_BLOCKS",
        "batch_size_pf",
        "num_m_blocks_pf",
        "num_n_blocks_pf",
        "high_load_wgs_pf",
        "max_tiles_per_wg_pf",
        "tiles_per_head_pf",
        "num_splits_pf",
        "prefill_ratio",
        "decode_ratio",
        "gqa_group_size",
        "total_programs_half",
    ],
)


@triton.jit(repr=_pod_persistent_repr)
def pod_persistent(
    # Prefill/Decode Communication
    cu_ctr,
    # Decode tensors
    Q, K, V,
    Mp, Lp, Op, Out,
    batch_num_block_n, locks,
    # Decode strides
    stride_qm, stride_qh, stride_qk,
    stride_kn, stride_kh, stride_kk,
    stride_vn, stride_vh, stride_vk,
    stride_om, stride_oh, stride_on,
    n_ctx_q_rows,  # decode N_CTX_Q (typically 1)
    stride_oph, stride_opm, stride_opn,
    sm_scale,  # softmax scale = 1/sqrt(head_dim)
    # Prefill tensors
    Q_pf, K_pf, V_pf,
    Mp_pf, Lp_pf, Op_pf, Out_pf,
    batch_num_block_n_pf, locks_pf,
    # Prefill strides
    stride_qm_pf, stride_qh_pf, stride_qk_pf,
    stride_kn_pf, stride_kh_pf, stride_kk_pf,
    stride_vn_pf, stride_vh_pf, stride_vk_pf,
    stride_om_pf, stride_oh_pf, stride_on_pf,
    n_ctx_q_rows_pf,  # prefill N_CTX_Q
    stride_oph_pf, stride_opm_pf, stride_opn_pf,
    # Constexpr — shared
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_ORIG: tl.constexpr,
    # Constexpr — decode
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MASKED_BLOCKS_DEC: tl.constexpr,
    batch_size: tl.constexpr,
    num_m_blocks: tl.constexpr,
    num_n_blocks: tl.constexpr,
    high_load_wgs: tl.constexpr,
    max_tiles_per_wg: tl.constexpr,
    tiles_per_head: tl.constexpr,
    num_splits: tl.constexpr,
    # Constexpr — prefill
    BLOCK_M_pf: tl.constexpr,
    BLOCK_N_pf: tl.constexpr,
    MASKED_BLOCKS: tl.constexpr,
    batch_size_pf: tl.constexpr,
    num_m_blocks_pf: tl.constexpr,
    num_n_blocks_pf: tl.constexpr,
    high_load_wgs_pf: tl.constexpr,
    max_tiles_per_wg_pf: tl.constexpr,
    tiles_per_head_pf: tl.constexpr,
    num_splits_pf: tl.constexpr,
    # Constexpr — common
    prefill_ratio: tl.constexpr,
    decode_ratio: tl.constexpr,
    max_output_tile_cnt: tl.constexpr,
    gqa_group_size: tl.constexpr,
    total_programs_half: tl.constexpr,
):
    # Deterministic work assignment based on program_id:
    # First half (0..total_programs_half-1) = decode
    # Second half (total_programs_half..2*total_programs_half-1) = prefill
    pid = tl.program_id(0)
    op = 0  # 0 = decode
    if pid >= total_programs_half:
        op = 1  # 1 = prefill

    current_pid = pid % total_programs_half

    if op == 0:  # decode
        module.la_persistent(
            True,           # is_pod
            current_pid,    # pod_pid
            Q, K, V,
            Mp, Lp, Op, Out,
            batch_num_block_n, locks,
            stride_qm, stride_qh, stride_qk,
            stride_kn, stride_kh, stride_kk,
            stride_vn, stride_vh, stride_vk,
            stride_om, stride_oh, stride_on,
            n_ctx_q_rows,
            stride_oph, stride_opm, stride_opn,
            sm_scale,
            # constexpr
            HEADS_PER_XCD=gqa_group_size,  # no XCD remap, all heads on one logical XCD
            HEAD_DIM_ORIG=HEAD_DIM_ORIG,
            HEAD_DIM=HEAD_DIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            MASKED_BLOCKS=MASKED_BLOCKS_DEC,
            XCD_REMAP=False,
            NUM_XCDS=1,
            batch_size=batch_size,
            causal=False,
            num_m_blocks=num_m_blocks,
            num_n_blocks=num_n_blocks,
            total_programs=total_programs_half,
            high_load_wgs=high_load_wgs,
            max_tiles_per_wg=max_tiles_per_wg,
            tiles_per_head=tiles_per_head,
            num_splits=num_splits,
            max_output_tile_cnt=max_output_tile_cnt,
            gqa_group_size=gqa_group_size,
            use_64_indexing=False,
            RAGGED_BATCH=False,
        )
    else:  # prefill
        module.la_persistent(
            True,           # is_pod
            current_pid,    # pod_pid
            Q_pf, K_pf, V_pf,
            Mp_pf, Lp_pf, Op_pf, Out_pf,
            batch_num_block_n_pf, locks_pf,
            stride_qm_pf, stride_qh_pf, stride_qk_pf,
            stride_kn_pf, stride_kh_pf, stride_kk_pf,
            stride_vn_pf, stride_vh_pf, stride_vk_pf,
            stride_om_pf, stride_oh_pf, stride_on_pf,
            n_ctx_q_rows_pf,
            stride_oph_pf, stride_opm_pf, stride_opn_pf,
            sm_scale,
            # constexpr
            HEADS_PER_XCD=gqa_group_size,
            HEAD_DIM_ORIG=HEAD_DIM_ORIG,
            HEAD_DIM=HEAD_DIM,
            BLOCK_M=BLOCK_M_pf,
            BLOCK_N=BLOCK_N_pf,
            MASKED_BLOCKS=MASKED_BLOCKS,
            XCD_REMAP=False,
            NUM_XCDS=1,
            batch_size=batch_size_pf,
            causal=True,
            num_m_blocks=num_m_blocks_pf,
            num_n_blocks=num_n_blocks_pf,
            total_programs=total_programs_half,
            high_load_wgs=high_load_wgs_pf,
            max_tiles_per_wg=max_tiles_per_wg_pf,
            tiles_per_head=tiles_per_head_pf,
            num_splits=num_splits_pf,
            max_output_tile_cnt=max_output_tile_cnt,
            gqa_group_size=gqa_group_size,
            use_64_indexing=False,
            RAGGED_BATCH=False,
        )
