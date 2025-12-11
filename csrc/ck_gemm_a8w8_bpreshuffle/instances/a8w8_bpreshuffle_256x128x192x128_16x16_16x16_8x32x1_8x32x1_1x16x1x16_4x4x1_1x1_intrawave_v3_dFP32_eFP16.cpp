// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "impl/a8w8_bpreshuffle_256x128x192x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_4x4x1_1x1_intrawave_v3.cuh"

template torch::Tensor
a8w8_bpreshuffle_256x128x192x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_4x4x1_1x1_intrawave_v3<F32, F16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y
    );

