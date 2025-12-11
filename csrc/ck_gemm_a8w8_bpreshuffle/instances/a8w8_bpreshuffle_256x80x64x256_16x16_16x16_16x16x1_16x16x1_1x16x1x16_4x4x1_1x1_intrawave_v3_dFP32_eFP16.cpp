// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "impl/a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3.cuh"

template torch::Tensor
a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3<F32, F16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y
    );

