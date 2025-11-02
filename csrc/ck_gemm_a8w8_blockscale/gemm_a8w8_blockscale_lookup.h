#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(DTYPE, ETYPE)                                                                                      \
   {                                                                                                                             \
       {0,                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x128_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {1,                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x64x128_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {2,                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {3,                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x128_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {4,                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x256x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {5,                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {6,                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {7,                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1<DTYPE, ETYPE>},                       \
       {8,                                                                                                       \
        a8w8_blockscale_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {9,                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x256x128_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {10,                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x128_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {11,                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {12,                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x128x256_16x16_32x32_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {13,                                                                                                       \
        a8w8_blockscale_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {14,                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x256x128_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {15,                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x128_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {16,                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x128_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {17,                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x128x256_16x16_32x32_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {18,                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x64x256_16x16_32x32_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {19,                                                                                                       \
        a8w8_blockscale_1x128x128_256x256x128x128_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {20,                                                                                                       \
        a8w8_blockscale_1x128x128_384x192x128x128_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {21,                                                                                                       \
        a8w8_blockscale_1x128x128_1024x256x256x128_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {22,                                                                                                       \
        a8w8_blockscale_1x128x128_1024x512x128x128_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {23,                                                                                                       \
        a8w8_blockscale_1x128x128_256x128x128x256_16x16_32x32_8x32x1_8x32x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
       {24,                                                                                                       \
        a8w8_blockscale_1x128x128_256x64x256x256_16x16_32x32_16x16x1_16x16x1_1x32x1x8_8_1x1_intrawave_v1<DTYPE, ETYPE>},                       \
   }

#endif // USE_ROCM
