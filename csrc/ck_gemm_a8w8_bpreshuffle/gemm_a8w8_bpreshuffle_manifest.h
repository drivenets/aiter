#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#include <cstdlib>

#include <torch/extension.h>

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x128_16x16_16x16_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x64_16x16_16x16_4x64x1_4x64x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x256x64_16x16_16x16_4x32x1_4x64x1_1x32x1x8_8x8x1_2x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x256x64_16x16_16x16_4x64x1_4x64x1_1x32x1x8_8x8x1_2x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x256x64_16x16_16x16_4x16x1_4x32x1_1x16x1x8_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x128x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v2(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x512x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x128x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v2(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x32x16x512_16x16_16x16_32x4x1_32x4x1_1x32x1x4_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x512x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x128x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x512x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x128x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x128x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x256x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v2(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v2(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x128_16x16_16x16_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x112x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x32x16x512_16x16_16x16_32x4x1_32x4x1_1x32x1x4_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v2(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x32x16x512_16x16_16x16_32x4x1_32x4x1_1x32x1x4_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v2(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x112x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x16x128x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x112x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x112x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x112x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x112x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x112x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x80x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x80x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x112x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x112x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x112x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x112x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x112x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x112x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);

template <typename DDataType, typename EDataType>
torch::Tensor
a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3(
    torch::Tensor &XQ,
    torch::Tensor &WQ,
    torch::Tensor &x_scale,
    torch::Tensor &w_scale,
    torch::Tensor &Y);


#endif // USE_ROCM
