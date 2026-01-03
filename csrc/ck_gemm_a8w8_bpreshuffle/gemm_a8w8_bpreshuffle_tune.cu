// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_bpreshuffle_common.cuh"
#include "gemm_a8w8_bpreshuffle_manifest.h"
#include "py_itfs_common.h"
#include <string>
#include <vector>

using BlockwiseKernel = std::function<torch::Tensor(
    torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&)>;

// Vector of all kernels indexed by kernel ID for tuning
using BlockwiseKernelVector = std::vector<BlockwiseKernel>;

// Helper function to return the next largest power of 2
static constexpr int nextPow2(unsigned int num)
{
    if(num <= 1)
        return 1;
    return 1 << (CHAR_BIT * sizeof(num) - __builtin_clz(num - 1));
}

// Macro to generate the kernel vector for tuning - each kernel gets an ID based on position
#define KERNEL_ENTRY(DTYPE, ETYPE, NAME) NAME<DTYPE, ETYPE>

#define GENERATE_KERNEL_VECTOR(DTYPE, ETYPE) { \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_128x16x256x64_16x16_16x16_4x16x1_4x32x1_1x16x1x8_8x8x1_1x2_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_128x16x32x128_16x16_16x16_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_128x16x32x128_16x16_16x16_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_128x32x16x512_16x16_16x16_32x4x1_32x4x1_1x32x1x4_4x4x1_1x1_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x112x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x112x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x112x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x128x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x128x128x64_16x16_16x16_4x64x1_4x64x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_4x4x1_1x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x128x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x160x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x16x128x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x16x128x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v2), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x16x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x16x256x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x16x512x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v2), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v2), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_4x4x1_1x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_4x4x1_1x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x32x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x32x256x64_16x16_16x16_4x32x1_4x64x1_1x32x1x8_8x8x1_2x1_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v2), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x64x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x64x256x64_16x16_16x16_4x64x1_4x64x1_1x32x1x8_8x8x1_2x1_intrawave_v1), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x80x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x96x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x96x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x96x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
    KERNEL_ENTRY(DTYPE, ETYPE, a8w8_bpreshuffle_256x96x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3), \
}

template <typename DDataType, typename EDataType = DDataType>
BlockwiseKernel blockwise_dispatch(int id)
{
    // Get kernel by ID from the vector
    static const BlockwiseKernelVector kernels = [] {
        if constexpr(std::is_same_v<EDataType, F16>)
        {
            return BlockwiseKernelVector GENERATE_KERNEL_VECTOR(DDataType, F16);
        }
        else if constexpr(std::is_same_v<EDataType, B16>)
        {
            return BlockwiseKernelVector GENERATE_KERNEL_VECTOR(DDataType, B16);
        }
        else
        {
            static_assert(false, "blockwise_dispatch used with unsupported dtype!");
        }
    }();

    TORCH_CHECK(id >= 0 && id < static_cast<int>(kernels.size()), 
                "Kernel id " + std::to_string(id) + " is out of range! Max: " + std::to_string(kernels.size() - 1));
    return kernels[id];
}

torch::Tensor gemm_a8w8_bpreshuffle_tune(torch::Tensor& XQ,
                                         torch::Tensor& WQ,
                                         torch::Tensor& x_scale,
                                         torch::Tensor& w_scale,
                                         torch::Tensor& Y,
                                         int kernelId,
                                         int splitK)
{
    TORCH_CHECK(XQ.dtype() == torch_fp8 && XQ.dtype() == WQ.dtype(),
                "Weights and activations should both be fp8!");
    TORCH_CHECK(x_scale.dtype() == w_scale.dtype(), "Scales should have the same dtype!");
    std::optional<torch::Tensor> bias = std::nullopt;

    int M      = XQ.size(0);
    int N      = WQ.size(0);
    int K      = XQ.size(1);
    int KBatch = std::pow(2, splitK);

    //if(Y.dtype() == at::ScalarType::Half)
    //{
    //    blockwise_dispatch<F32, F16>(kernelId)(XQ, WQ, x_scale, w_scale, Y);
    //}
    if (Y.dtype() == at::ScalarType::BFloat16)
    {
        blockwise_dispatch<F32, B16>(kernelId)(XQ, WQ, x_scale, w_scale, Y);
    } else
    {
        TORCH_CHECK(false, "Unsupported scales/output dtype!");
    }
    return Y;
}
