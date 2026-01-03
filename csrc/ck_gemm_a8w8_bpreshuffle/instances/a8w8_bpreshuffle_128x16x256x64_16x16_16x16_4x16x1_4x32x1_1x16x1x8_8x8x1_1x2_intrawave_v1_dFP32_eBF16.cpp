// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#include "impl/a8w8_bpreshuffle_128x16x256x64_16x16_16x16_4x16x1_4x32x1_1x16x1x8_8x8x1_1x2_intrawave_v1.cuh"

template torch::Tensor
a8w8_bpreshuffle_128x16x256x64_16x16_16x16_4x16x1_4x32x1_1x16x1x8_8x8x1_1x2_intrawave_v1<F32, B16>(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y
    );

