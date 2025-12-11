// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_bpreshuffle_common.cuh"

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x144x64x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
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
    bool pad = (M % 144 != 0) || (N % 64 != 0) || (K % (128) != 0);
    if (pad)
    {
        // pad
        using DeviceGemmInstance = DeviceGemmHelperF8Flatmm<
            DDataType, EDataType,
            256,
            144, 64, 128,
            8, 16,
            16, 16,
            9, 1,
            S<16, 16, 1>,
            S<8, 32, 1>,
            1,
            1,
            S<1, 16, 1, 16>,
            S<4, 4, 1>,
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
            144, 64, 128,
            8, 16,
            16, 16,
            9, 1,
            S<16, 16, 1>,
            S<8, 32, 1>,
            1,
            1,
            S<1, 16, 1, 16>,
            S<4, 4, 1>,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
            ck::tensor_operation::device::GemmSpecialization::Default>;
        // Run kernel instance.
        return gemm_a8w8_bpreshuffle_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);

        // no pad
    }
}

