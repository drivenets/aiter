// SPDX-License-Identifier: MIT
// MXFP4 MoE GEMV kernel for decode (small M).
// Processes unshuffled, minimally-padded weights to avoid CK-tile 256-alignment waste.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <cstdint>

using bf16_t = hip_bfloat16;

__device__ __forceinline__ float bf16_to_float(bf16_t v) {
    uint32_t bits = (uint32_t)v.data << 16;
    return __uint_as_float(bits);
}

__device__ __constant__ float fp4_lut[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

__device__ __forceinline__ float e8m0_to_float(uint8_t e) {
    uint32_t bits = (uint32_t)e << 23;
    return __uint_as_float(bits);
}

// V1: One thread per N, shared memory for activation.
// Grid: (num_sorted, ceil(N/BLOCK_N))
template <int BLOCK_N>
__global__ void mxfp4_moe_gemv_v1(
    const bf16_t* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ sorted_expert_ids,
    const int32_t* __restrict__ num_valid_ids,
    const float* __restrict__ sorted_weights,
    float* __restrict__ output,
    int K_packed, int K_actual, int K_groups,
    int N, int N_out, int max_token_id,
    int64_t stride_x, int64_t stride_w_e, int64_t stride_w_n,
    int64_t stride_s_e, int64_t stride_s_n,
    int block_m, int is_gemm2
) {
    const int pid_token = blockIdx.x;
    const int tid = threadIdx.x;

    const int num_valid = num_valid_ids[0];
    if (pid_token >= num_valid) return;

    const int32_t packed_id = sorted_ids[pid_token];
    const int token_id = packed_id & 0xFFFFFF;
    if (token_id >= max_token_id) return;

    const int block_idx = pid_token / block_m;
    const int expert_id = sorted_expert_ids[block_idx];
    const int n_idx = blockIdx.y * BLOCK_N + tid;

    // Load activation into dynamic shared memory
    extern __shared__ float x_sh[];
    const bf16_t* x_row = x + token_id * stride_x;
    for (int i = tid; i < K_actual; i += BLOCK_N) {
        x_sh[i] = bf16_to_float(x_row[i]);
    }
    __syncthreads();

    if (n_idx >= N) return;

    const uint8_t* w_row = weight + expert_id * stride_w_e + n_idx * stride_w_n;
    const uint8_t* s_row = scale + expert_id * stride_s_e + n_idx * stride_s_n;

    float acc = 0.0f;
    for (int g = 0; g < K_groups; g++) {
        const float s = e8m0_to_float(s_row[g]);
        const int k_base = g * 32;
        const uint8_t* w_grp = w_row + g * 16;
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            uint8_t wb = w_grp[j];
            int k = k_base + j * 2;
            acc += fp4_lut[wb & 0xF] * s * x_sh[k];
            acc += fp4_lut[(wb >> 4) & 0xF] * s * x_sh[k + 1];
        }
    }

    if (is_gemm2) acc *= sorted_weights[pid_token];
    atomicAdd(&output[token_id * N_out + n_idx], acc);
}


// V2: Multiple N rows per thread. Handles all N in fewer threads.
// Grid: (num_sorted, ceil(N / (BLOCK_N * N_PER_THREAD)))
template <int BLOCK_N, int N_PER_THREAD>
__global__ void mxfp4_moe_gemv_v2(
    const bf16_t* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ sorted_expert_ids,
    const int32_t* __restrict__ num_valid_ids,
    const float* __restrict__ sorted_weights,
    float* __restrict__ output,
    int K_packed, int K_actual, int K_groups,
    int N, int N_out, int max_token_id,
    int64_t stride_x, int64_t stride_w_e, int64_t stride_w_n,
    int64_t stride_s_e, int64_t stride_s_n,
    int block_m, int is_gemm2
) {
    const int pid_token = blockIdx.x;
    const int tid = threadIdx.x;

    const int num_valid = num_valid_ids[0];
    if (pid_token >= num_valid) return;

    const int32_t packed_id = sorted_ids[pid_token];
    const int token_id = packed_id & 0xFFFFFF;
    if (token_id >= max_token_id) return;

    const int block_idx = pid_token / block_m;
    const int expert_id = sorted_expert_ids[block_idx];

    extern __shared__ float x_sh[];
    const bf16_t* x_row = x + token_id * stride_x;
    for (int i = tid; i < K_actual; i += BLOCK_N) {
        x_sh[i] = bf16_to_float(x_row[i]);
    }
    __syncthreads();

    const int n_base = blockIdx.y * BLOCK_N * N_PER_THREAD + tid;
    float routing_w = is_gemm2 ? sorted_weights[pid_token] : 1.0f;

    #pragma unroll
    for (int nn = 0; nn < N_PER_THREAD; nn++) {
        const int n_idx = n_base + nn * BLOCK_N;
        if (n_idx >= N) break;

        const uint8_t* w_row = weight + expert_id * stride_w_e + n_idx * stride_w_n;
        const uint8_t* s_row = scale + expert_id * stride_s_e + n_idx * stride_s_n;

        float acc = 0.0f;
        for (int g = 0; g < K_groups; g++) {
            const float s = e8m0_to_float(s_row[g]);
            const int k_base = g * 32;
            const uint8_t* w_grp = w_row + g * 16;
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                uint8_t wb = w_grp[j];
                int k = k_base + j * 2;
                acc += fp4_lut[wb & 0xF] * s * x_sh[k];
                acc += fp4_lut[(wb >> 4) & 0xF] * s * x_sh[k + 1];
            }
        }

        atomicAdd(&output[token_id * N_out + n_idx], acc * routing_w);
    }
}


// Launchers
extern "C" void launch_mxfp4_moe_gemv(
    const void* x, const void* weight, const void* scale,
    const void* sorted_ids, const void* sorted_expert_ids,
    const void* num_valid_ids, const void* sorted_weights,
    void* output,
    int K_packed, int K_groups, int N, int N_out, int max_token_id,
    int64_t stride_x, int64_t stride_w_e, int64_t stride_w_n,
    int64_t stride_s_e, int64_t stride_s_n,
    int block_m, int is_gemm2,
    int num_sorted_tokens,
    int block_n,  // BLOCK_N to use
    hipStream_t stream
) {
    int K_actual = K_packed * 2;
    int shmem_bytes = K_actual * sizeof(float);

    #define LAUNCH_V1(BN) do { \
        dim3 grid(num_sorted_tokens, (N + BN - 1) / BN); \
        mxfp4_moe_gemv_v1<BN><<<grid, BN, shmem_bytes, stream>>>( \
            (const bf16_t*)x, (const uint8_t*)weight, (const uint8_t*)scale, \
            (const int32_t*)sorted_ids, (const int32_t*)sorted_expert_ids, \
            (const int32_t*)num_valid_ids, (const float*)sorted_weights, \
            (float*)output, K_packed, K_actual, K_groups, N, N_out, max_token_id, \
            stride_x, stride_w_e, stride_w_n, stride_s_e, stride_s_n, \
            block_m, is_gemm2); \
    } while(0)

    #define LAUNCH_V2(BN, NPT) do { \
        dim3 grid(num_sorted_tokens, (N + BN * NPT - 1) / (BN * NPT)); \
        mxfp4_moe_gemv_v2<BN, NPT><<<grid, BN, shmem_bytes, stream>>>( \
            (const bf16_t*)x, (const uint8_t*)weight, (const uint8_t*)scale, \
            (const int32_t*)sorted_ids, (const int32_t*)sorted_expert_ids, \
            (const int32_t*)num_valid_ids, (const float*)sorted_weights, \
            (float*)output, K_packed, K_actual, K_groups, N, N_out, max_token_id, \
            stride_x, stride_w_e, stride_w_n, stride_s_e, stride_s_n, \
            block_m, is_gemm2); \
    } while(0)

    switch (block_n) {
        case 256:  LAUNCH_V1(256); break;
        case 512:  LAUNCH_V1(512); break;
        case 1024: LAUNCH_V1(1024); break;
        // V2 variants: (threads, N_per_thread) → effective block_n
        case 2048: LAUNCH_V2(256, 8); break;   // 256 threads × 8 N/thread
        case 4096: LAUNCH_V2(256, 16); break;  // 256 threads × 16 N/thread
        default:   LAUNCH_V1(256); break;
    }
}
