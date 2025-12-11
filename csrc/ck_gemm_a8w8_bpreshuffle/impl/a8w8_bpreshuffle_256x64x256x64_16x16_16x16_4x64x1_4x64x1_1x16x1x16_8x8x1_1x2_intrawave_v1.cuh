// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_bpreshuffle_common.cuh"

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x256x64_16x16_16x16_4x64x1_4x64x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
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
    bool pad = (M % 64 != 0) || (N % 256 != 0) || (K % (64) != 0);
    if (pad)
    {
        // pad
        using DeviceGemmInstance = DeviceGemmHelperF8Flatmm<
            DDataType, EDataType,
            256,
            64, 256, 64,
            16, 16,
            16, 16,
            4, 4,
            S<4, 64, 1>,
            S<4, 64, 1>,
            1,
            2,
            S<1, 16, 1, 16>,
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
            64, 256, 64,
            16, 16,
            16, 16,
            4, 4,
            S<4, 64, 1>,
            S<4, 64, 1>,
            1,
            2,
            S<1, 16, 1, 16>,
            S<8, 8, 1>,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v1,
            ck::tensor_operation::device::GemmSpecialization::Default>;
        // Run kernel instance.
        return gemm_a8w8_bpreshuffle_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);

        // no pad
    }
}

