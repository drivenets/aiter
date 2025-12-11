// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "impl/a8w8_bpreshuffle_256x64x16x512_16x16_16x16_32x8x1_32x8x1_1x64x1x4_4x4x1_1x1_intrawave_v2.cuh"

template torch::Tensor
a8w8_bpreshuffle_256x64x16x512_16x16_16x16_32x8x1_32x8x1_1x64x1x4_4x4x1_1x1_intrawave_v2<F32, F16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y
    );

