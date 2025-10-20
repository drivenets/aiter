#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(DTYPE, ETYPE)                                                                                      \
   {                                                                                                                             \
       {{1, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<DTYPE, ETYPE>},                       \
       {{1, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x128x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v2<DTYPE, ETYPE>},                       \
       {{1, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x512x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x16x128x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v2<DTYPE, ETYPE>},                       \
       {{7, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{7, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x32x16x512_16x16_16x16_32x4x1_32x4x1_1x32x1x4_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{7, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x512x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{7, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x16x128x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x512x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x16x128x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x16x128x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x256x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v2<DTYPE, ETYPE>},                       \
       {{64, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v2<DTYPE, ETYPE>},                       \
       {{128, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{128, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1, 1024, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 1024, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 1024, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 1024, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 1024, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 1024, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 1024, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 1024, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 1024, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{512, 1024, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 28672, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x128_16x16_16x16_8x16x1_8x16x1_1x16x1x8_4x4x1_1x1_intrawave_v2<DTYPE, ETYPE>},                       \
       {{2, 28672, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 28672, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 28672, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 28672, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 28672, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 28672, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 28672, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x64x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 28672, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 28672, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{768, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x112x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x96x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{768, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{768, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{768, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x96x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{7, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{7, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{7, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<DTYPE, ETYPE>},                       \
       {{16, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{16, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x32x16x512_16x16_16x16_32x4x1_32x4x1_1x32x1x4_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{32, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v2<DTYPE, ETYPE>},                       \
       {{64, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{64, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{128, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{256, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x128x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{512, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1024, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1024, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{2, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<DTYPE, ETYPE>},                       \
       {{4, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{4, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{8, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{12, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{12, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{12, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{24, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{24, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{24, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{40, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{40, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{40, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v2<DTYPE, ETYPE>},                       \
       {{48, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{48, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{56, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{56, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{56, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{72, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{72, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{72, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{80, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{80, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{80, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{88, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{88, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{88, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{96, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x32x16x512_16x16_16x16_32x4x1_32x4x1_1x32x1x4_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{96, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v2<DTYPE, ETYPE>},                       \
       {{96, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{104, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{104, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{104, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{112, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{112, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{112, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{120, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{120, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{120, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{768, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{768, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{768, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1536, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1536, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1536, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2048, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{3904, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{3904, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{3904, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{3904, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{36288, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{36288, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{36288, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{36288, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{42628, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{42628, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{42628, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{42628, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65456, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65456, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65456, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65456, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65490, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65490, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65490, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{65490, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{136, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{136, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{136, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{144, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{144, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{144, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{152, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{152, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{152, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{160, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{160, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{168, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{168, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{168, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{176, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{176, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{176, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{184, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{184, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{184, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{192, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{200, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{200, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{200, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{208, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{208, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{208, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{216, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{216, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{216, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{224, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{224, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{224, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x112x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{232, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{232, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{232, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{240, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{240, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{240, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{248, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{248, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{248, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{272, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_128x16x32x512_16x16_16x16_32x4x1_32x4x1_1x16x1x8_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{272, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{272, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{288, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{288, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{304, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{304, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{304, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{320, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x64x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{320, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{320, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{336, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{336, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{336, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{352, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{352, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{352, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{368, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{368, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{368, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{384, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x512_16x16_16x16_32x8x1_32x8x1_1x32x1x8_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{384, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{384, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{400, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x16x128x512_16x16_16x16_32x8x1_32x8x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{400, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{400, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x112x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{416, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{416, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{416, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x112x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{432, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{432, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{432, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x112x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{448, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x112x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{448, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x112x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{464, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{464, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{464, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{480, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{480, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{496, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{496, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{496, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{544, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{544, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x80x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{544, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{576, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{576, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{608, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{608, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x80x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{608, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{640, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{640, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{640, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{672, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{672, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{672, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{704, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{704, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{704, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{736, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{736, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{736, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{800, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{800, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{800, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x112x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{832, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{832, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x112x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{832, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x112x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{864, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{864, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{864, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x112x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{896, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{896, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{896, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x112x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{928, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{928, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{928, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{960, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{960, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x160x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{960, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{992, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{992, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{992, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1056, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1056, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1056, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1088, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1088, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1088, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1120, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x48x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1120, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1120, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1152, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1152, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1152, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1184, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x48x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1184, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1184, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1216, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1216, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1216, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1248, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1248, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1248, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1280, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1280, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1280, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1312, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x32x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {{1312, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1312, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1344, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x112x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1344, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1344, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1376, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1376, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1376, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1408, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1408, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1408, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1440, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1440, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1440, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1472, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1472, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1472, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1504, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1504, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1504, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1568, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1568, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1568, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1600, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1600, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1600, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1632, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1632, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1632, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1664, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1664, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1664, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1696, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1696, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1696, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1728, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1728, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1728, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1760, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x80x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4x4x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1760, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1760, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1792, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1792, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1792, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1824, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1824, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1824, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1856, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1856, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1856, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1888, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1888, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1888, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1920, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1920, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1920, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1952, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1952, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1952, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1984, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1984, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x160x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{1984, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x128x256x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2016, 1280, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2016, 8192, 1024},                                                                                                       \
        a8w8_bpreshuffle_256x96x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{2016, 8192, 4096},                                                                                                       \
        a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
   }

#endif // USE_ROCM
