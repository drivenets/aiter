// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "gemm_a8w8_blockscale_common.cuh"

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3(
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
    bool pad = (M % 128 != 0) || (N % 128 != 0) || (K % (128) != 0);
    if (pad)
    {
        // pad
        using DeviceGemmInstance = DeviceGemmHelperF8BlockScale<
            DDataType, EDataType,
            256,
            1, 128, 128,
            128, 128, 128,
            16, 16,
            32, 32,
            2, 2,
            S<8, 32, 1>,
            S<8, 32, 1>,
            1,
            1,
            S<1, 32, 1, 8>,
            S<8>,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
            ck::tensor_operation::device::GemmSpecialization::MNKPadding>;
        // Run kernel instance.
        return gemm_a8w8_blockscale_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);

        // pad
    }
    else
    {
        // no pad
        using DeviceGemmInstance = DeviceGemmHelperF8BlockScale<
            DDataType, EDataType,
            256,
            1, 128, 128,
            128, 128, 128,
            16, 16,
            32, 32,
            2, 2,
            S<8, 32, 1>,
            S<8, 32, 1>,
            1,
            1,
            S<1, 32, 1, 8>,
            S<8>,
            ck::BlockGemmPipelineScheduler::Intrawave,
            ck::BlockGemmPipelineVersion::v3,
            ck::tensor_operation::device::GemmSpecialization::Default>;
        // Run kernel instance.
        return gemm_a8w8_blockscale_impl<DDataType, EDataType, DeviceGemmInstance>(XQ, WQ, x_scale, w_scale, Y);

        // no pad
    }
}

