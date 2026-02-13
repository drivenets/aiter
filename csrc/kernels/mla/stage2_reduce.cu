// MLA decode stage2 reduce kernel - HIP C++ replacement for Triton _fwd_kernel_stage2_asm
// Combines partial attention outputs from stage1 KV-split using log-sum-exp weighted reduction.
//
// For each (batch, head): reduce num_valid_kv_splits partial outputs into final output
//   acc = sum_i( exp(lse_i - max_lse) * partial_output_i )
//   output = acc / sum_i( exp(lse_i - max_lse) )

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPStream.h>
#include <type_traits>

template <typename OutT, int Lv>
__global__ void mla_stage2_reduce_kernel(
    const float* __restrict__ Mid_O,
    const float* __restrict__ Mid_lse,
    OutT* __restrict__ O,
    const int32_t* __restrict__ qo_indptr,
    const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ num_kv_splits_indptr,
    int64_t stride_mid_ob,
    int64_t stride_mid_oh,
    int64_t stride_mid_os,
    int64_t stride_obs,
    int64_t stride_oh,
    int mgc,
    int batch_num
) {
    const int cur_batch = blockIdx.x;
    const int cur_head = blockIdx.y;
    const int tid = threadIdx.x;

    const int cur_qo_start = qo_indptr[cur_batch];
    const int cur_qo_end = qo_indptr[cur_batch + 1];
    const int cur_split_start = num_kv_splits_indptr[cur_batch];
    const int cur_split_end = num_kv_splits_indptr[cur_batch + 1];
    const int cur_kv_seq_len = kv_indptr[cur_batch + 1] - kv_indptr[cur_batch];

    const int num_valid_kv_splits = min(
        cur_split_end - cur_split_start,
        (cur_kv_seq_len + mgc - 1) / mgc
    );

    for (int cur_qo = cur_qo_start; cur_qo < cur_qo_end; cur_qo++) {
        const int64_t base_logic = (int64_t)cur_qo * stride_mid_ob + (int64_t)cur_head * stride_mid_oh;

        for (int d = tid; d < Lv; d += blockDim.x) {
            float e_sum = 0.0f;
            float e_max = -INFINITY;
            float acc = 0.0f;

            for (int split_id = 0; split_id < num_valid_kv_splits; split_id++) {
                const int64_t split_offset = (int64_t)split_id * stride_mid_os;
                float tv = Mid_O[(base_logic + split_offset) * Lv + d];
                float tlogic = Mid_lse[base_logic + split_offset];

                float n_e_max = fmaxf(tlogic, e_max);
                float old_scale = expf(e_max - n_e_max);
                float exp_logic = expf(tlogic - n_e_max);
                acc = acc * old_scale + exp_logic * tv;
                e_sum = e_sum * old_scale + exp_logic;
                e_max = n_e_max;
            }

            float result = acc / e_sum;
            // Use __float2bfloat16 for bf16 output, direct assignment for float
            if constexpr (std::is_same_v<OutT, __hip_bfloat16>) {
                O[(int64_t)cur_qo * stride_obs + (int64_t)cur_head * stride_oh + d] = __float2bfloat16(result);
            } else if constexpr (std::is_same_v<OutT, __half>) {
                O[(int64_t)cur_qo * stride_obs + (int64_t)cur_head * stride_oh + d] = __float2half(result);
            } else {
                O[(int64_t)cur_qo * stride_obs + (int64_t)cur_head * stride_oh + d] = result;
            }
        }
    }
}

void mla_stage2_reduce(
    torch::Tensor Mid_O,
    torch::Tensor Mid_lse,
    torch::Tensor O,
    torch::Tensor qo_indptr,
    torch::Tensor kv_indptr,
    torch::Tensor num_kv_splits_indptr,
    int64_t stride_mid_ob,
    int64_t stride_mid_oh,
    int64_t stride_mid_os,
    int64_t stride_obs,
    int64_t stride_oh,
    int mgc,
    int batch_num,
    int bs,
    int nhead,
    int v_head_dim
) {
    dim3 grid(bs, nhead);
    int block_size = (v_head_dim <= 128) ? 128 : (v_head_dim <= 256) ? 256 : 512;
    dim3 block(block_size);
    auto stream = c10::hip::getCurrentHIPStream().stream();

    // Dispatch on output dtype and v_head_dim
    #define LAUNCH(OutT, LV_VAL) \
        mla_stage2_reduce_kernel<OutT, LV_VAL><<<grid, block, 0, stream>>>( \
            Mid_O.data_ptr<float>(), Mid_lse.data_ptr<float>(), \
            reinterpret_cast<OutT*>(O.data_ptr()), \
            qo_indptr.data_ptr<int32_t>(), kv_indptr.data_ptr<int32_t>(), \
            num_kv_splits_indptr.data_ptr<int32_t>(), \
            stride_mid_ob, stride_mid_oh, stride_mid_os, \
            stride_obs, stride_oh, mgc, batch_num)

    #define DISPATCH_LV(OutT) \
        if (v_head_dim <= 128) { LAUNCH(OutT, 128); } \
        else if (v_head_dim <= 256) { LAUNCH(OutT, 256); } \
        else if (v_head_dim <= 512) { LAUNCH(OutT, 512); } \
        else if (v_head_dim <= 576) { LAUNCH(OutT, 576); } \
        else { AT_ERROR("Unsupported v_head_dim: ", v_head_dim); }

    if (O.scalar_type() == at::ScalarType::Float) {
        DISPATCH_LV(float);
    } else if (O.scalar_type() == at::ScalarType::BFloat16) {
        DISPATCH_LV(__hip_bfloat16);
    } else if (O.scalar_type() == at::ScalarType::Half) {
        DISPATCH_LV(__half);
    } else {
        AT_ERROR("Unsupported output dtype: ", O.scalar_type());
    }

    #undef LAUNCH
    #undef DISPATCH_LV
}

PYBIND11_MODULE(mla_stage2_reduce_hip, m) {
    m.def("mla_stage2_reduce", &mla_stage2_reduce, "MLA stage2 reduce (HIP)");
}
