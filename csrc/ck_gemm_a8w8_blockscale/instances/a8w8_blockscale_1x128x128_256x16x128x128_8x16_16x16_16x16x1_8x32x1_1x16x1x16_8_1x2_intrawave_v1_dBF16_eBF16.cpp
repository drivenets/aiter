// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "impl/a8w8_blockscale_1x128x128_256x16x128x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_8_1x2_intrawave_v1.cuh"

template torch::Tensor
a8w8_blockscale_1x128x128_256x16x128x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_8_1x2_intrawave_v1<F32, B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y
    );

