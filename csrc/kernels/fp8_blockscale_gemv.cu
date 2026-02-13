// FP8 blockscale GEMV kernel for M=1 decode
// Replaces splitK GEMM + separate reduce with a single fused kernel
// Target: DeepSeek R1 attention projections on MI355X
//
// Weight layout: [N, K] in FP8 (e4m3fn), row-major
// Scale layout: [N/GROUP_N, K/GROUP_K] in FP32, where GROUP_N=128, GROUP_K=128
// Input: [K] in FP8 (e4m3fn) with per-group scales [K/GROUP_K] in FP32
// Output: [N] in BF16

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#include <hip/hip_fp8.h>

// Tuning parameters
// BLOCK_N: number of output elements per thread block
// THREADS: threads per block
// K_UNROLL: K elements processed per inner loop iteration

template <int BLOCK_N, int THREADS, int GROUP_K = 128, int GROUP_N = 128>
__global__ void __launch_bounds__(THREADS, 2)
fp8_blockscale_gemv_kernel(
    const __hip_fp8_e4m3_fnuz* __restrict__ weight,   // [N, K] row-major
    const __hip_fp8_e4m3_fnuz* __restrict__ input,     // [K]
    const float* __restrict__ w_scale,                  // [N/GROUP_N, K/GROUP_K]
    const float* __restrict__ input_scale,              // [K/GROUP_K]
    __hip_bfloat16* __restrict__ output,                // [N]
    const int N,
    const int K)
{
    const int n_base = blockIdx.x * BLOCK_N;
    const int tid = threadIdx.x;

    // Each thread accumulates BLOCK_N output elements
    // But BLOCK_N may be > THREADS, so we tile
    constexpr int N_PER_THREAD = (BLOCK_N + THREADS - 1) / THREADS;

    float acc[N_PER_THREAD];
    #pragma unroll
    for (int i = 0; i < N_PER_THREAD; i++)
        acc[i] = 0.0f;

    const int k_groups = K / GROUP_K;
    const int n_scale_stride = k_groups;  // w_scale is [N/GROUP_N, K/GROUP_K]

    // Shared memory for input tile (one K-group at a time)
    __shared__ float s_input[GROUP_K];      // dequantized input for current K-group
    __shared__ float s_input_scale;

    // Process K in groups of GROUP_K
    for (int kg = 0; kg < k_groups; kg++)
    {
        const int k_offset = kg * GROUP_K;

        // Collaboratively load and dequantize input for this K-group
        const float inp_scale = input_scale[kg];
        for (int i = tid; i < GROUP_K; i += THREADS)
        {
            float val = static_cast<float>(input[k_offset + i]);
            s_input[i] = val * inp_scale;
        }
        __syncthreads();

        // Each thread processes its assigned N elements
        #pragma unroll
        for (int ni = 0; ni < N_PER_THREAD; ni++)
        {
            const int n_idx = n_base + tid + ni * THREADS;
            if (n_idx >= N) continue;

            // Get weight scale for this (N-block, K-block)
            const int n_scale_idx = (n_idx / GROUP_N);
            const float ws = w_scale[n_scale_idx * n_scale_stride + kg];

            // Dot product over GROUP_K elements
            const __hip_fp8_e4m3_fnuz* w_row = weight + (size_t)n_idx * K + k_offset;

            float dot = 0.0f;
            // Process 8 elements at a time (128 bytes / 16 = 8 FP8 elements per vector load)
            for (int ki = 0; ki < GROUP_K; ki += 8)
            {
                // Load 8 FP8 weights
                const uint64_t* w_ptr = reinterpret_cast<const uint64_t*>(w_row + ki);
                uint64_t w_packed = *w_ptr;
                const __hip_fp8_e4m3_fnuz* w8 = reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(&w_packed);

                dot += static_cast<float>(w8[0]) * s_input[ki + 0];
                dot += static_cast<float>(w8[1]) * s_input[ki + 1];
                dot += static_cast<float>(w8[2]) * s_input[ki + 2];
                dot += static_cast<float>(w8[3]) * s_input[ki + 3];
                dot += static_cast<float>(w8[4]) * s_input[ki + 4];
                dot += static_cast<float>(w8[5]) * s_input[ki + 5];
                dot += static_cast<float>(w8[6]) * s_input[ki + 6];
                dot += static_cast<float>(w8[7]) * s_input[ki + 7];
            }

            acc[ni] += dot * ws;
        }

        __syncthreads();
    }

    // Write output
    #pragma unroll
    for (int ni = 0; ni < N_PER_THREAD; ni++)
    {
        const int n_idx = n_base + tid + ni * THREADS;
        if (n_idx < N)
        {
            output[n_idx] = __float2bfloat16(acc[ni]);
        }
    }
}

// Launcher
extern "C" void fp8_blockscale_gemv(
    const void* weight,
    const void* input,
    const void* w_scale,
    const void* input_scale,
    void* output,
    int N,
    int K,
    hipStream_t stream)
{
    // BLOCK_N=4, THREADS=256: each block handles 4 output elements
    // N_PER_THREAD = 1 when BLOCK_N <= THREADS (4 <= 256)
    // But we want more work per block. Let's use BLOCK_N=128, THREADS=256
    // N_PER_THREAD = 1 since each thread handles 1 N element (128 < 256, some threads idle)
    // Actually: BLOCK_N=256, THREADS=256, N_PER_THREAD=1

    constexpr int BLOCK_N = 256;
    constexpr int THREADS = 256;

    int num_blocks = (N + BLOCK_N - 1) / BLOCK_N;

    hipLaunchKernelGGL(
        (fp8_blockscale_gemv_kernel<BLOCK_N, THREADS>),
        dim3(num_blocks), dim3(THREADS), 0, stream,
        reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(weight),
        reinterpret_cast<const __hip_fp8_e4m3_fnuz*>(input),
        reinterpret_cast<const float*>(w_scale),
        reinterpret_cast<const float*>(input_scale),
        reinterpret_cast<__hip_bfloat16*>(output),
        N, K);
}
