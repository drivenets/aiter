// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "impl/a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3.cuh"

template torch::Tensor
a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<F32, B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y
    );

