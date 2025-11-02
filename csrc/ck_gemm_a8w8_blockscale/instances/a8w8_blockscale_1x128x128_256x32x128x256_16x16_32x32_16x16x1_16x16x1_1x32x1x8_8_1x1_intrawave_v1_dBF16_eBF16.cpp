// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "impl/a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1.cuh"

template torch::Tensor
a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<F32, B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y
    );

