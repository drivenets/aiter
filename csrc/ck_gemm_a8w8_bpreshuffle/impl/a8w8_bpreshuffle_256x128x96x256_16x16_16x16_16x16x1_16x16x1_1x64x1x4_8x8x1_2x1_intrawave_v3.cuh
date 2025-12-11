// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_bpreshuffle_common.cuh"

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x96x256_16x16_16x16_16x16x1_16x16x1_1x64x1x4_8x8x1_2x1_intrawave_v3(
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
    bool pad = (M % 128 != 0) || (N % 96 != 0) || (K % (256) != 0);
    if (pad)
    {
        // pad
        using DeviceGemmInstance = DeviceGemmHelperF8Flatmm<
            DDataType, EDataType,
            256,
            128, 96, 256,
            16, 16,
            16, 16,
            4, 3,
            S<16, 16, 1>,
            S<16, 16, 1>,
            2,
            1,
            S<1, 64, 1, 4>,
            S<8, 8, 1>,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
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
            128, 96, 256,
            16, 16,
            16, 16,
            4, 3,
            S<16, 16, 1>,
            S<16, 16, 1>,
            2,
            1,
            S<1, 64, 1, 4>,
            S<8, 8, 1>,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
            ck::tensor_operation::device::GemmSpecialization::Default>;
        // Run kernel instance.
        return gemm_a8w8_bpreshuffle_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);

        // no pad
    }
}

