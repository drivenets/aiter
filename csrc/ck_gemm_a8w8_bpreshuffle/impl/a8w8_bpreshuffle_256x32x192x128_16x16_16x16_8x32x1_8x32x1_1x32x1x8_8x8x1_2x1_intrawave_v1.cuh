// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_bpreshuffle_common.cuh"

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x192x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y
    )
{
    // The smallest kernel we have available. Works well for memory bound shapes.

    // Check if this input needs to be padded.
    int M = size_to_dim_(XQ.dim() - 1, XQ.sizes());
    int N = WQ.size(0);
    int K = WQ.size(1);
    bool pad = (M % 32 != 0) || (N % 192 != 0) || (K % (128) != 0);
    if (pad)
    {
        // pad
        using DeviceGemmInstance = DeviceGemmHelperF8Flatmm<
            DDataType, EDataType,
            256,
            32, 192, 128,
            16, 16,
            16, 16,
            2, 3,
            S<8, 32, 1>,
            S<8, 32, 1>,
            2,
            1,
            S<1, 32, 1, 8>,
            S<8, 8, 1>,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
        // Run kernel instance.
        return gemm_a8w8_bpreshuffle_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);

        // pad
    }
    else
    {
        // no pad
        using DeviceGemmInstance = DeviceGemmHelperF8Flatmm<
            DDataType, EDataType,
            256,
            32, 192, 128,
            16, 16,
            16, 16,
            2, 3,
            S<8, 32, 1>,
            S<8, 32, 1>,
            2,
            1,
            S<1, 32, 1, 8>,
            S<8, 8, 1>,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            ck::tensor_operation::device::GemmSpecialization::Default>;
        // Run kernel instance.
        return gemm_a8w8_bpreshuffle_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);

        // no pad
    }
}

