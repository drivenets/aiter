#pragma once
// SPDX-License-Identifier: MIT
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

#ifdef USE_ROCM

#define GENERATE_LOOKUP_TABLE(DTYPE, ETYPE)                                                                                      \
   {                                                                                                                             \
       {{16, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x128x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{16, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x64x16x512_16x16_16x16_32x8x1_32x8x1_1x64x1x4_4x4x1_1x1_intrawave_v2<DTYPE, ETYPE>},                       \
       {{32, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{32, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x64x16x512_16x16_16x16_32x8x1_32x8x1_1x64x1x4_4x4x1_1x1_intrawave_v2<DTYPE, ETYPE>},                       \
       {{48, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x64x16x512_16x16_16x16_32x8x1_32x8x1_1x64x1x4_4x4x1_1x1_intrawave_v2<DTYPE, ETYPE>},                       \
       {{48, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{48, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{48, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_1x2_intrawave_v3<DTYPE, ETYPE>},                       \
       {{64, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{64, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x64x16x512_16x16_16x16_32x8x1_32x8x1_1x64x1x4_4x4x1_1x1_intrawave_v2<DTYPE, ETYPE>},                       \
       {{64, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x192x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{64, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x64x16x512_16x16_16x16_32x8x1_32x8x1_1x64x1x4_4x4x1_1x1_intrawave_v2<DTYPE, ETYPE>},                       \
       {{80, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{80, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x224x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{80, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x160x224x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{80, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x224x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{96, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{96, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x224x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{96, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x160x224x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_4x4x1_1x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{96, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x224x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{112, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x224x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{112, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{112, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x224x128_16x16_16x16_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{112, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x224x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{128, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x224x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{128, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{128, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x224x128_16x16_16x16_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{128, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x256x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{144, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x224x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{144, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x224x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{144, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x224x128_16x16_16x16_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{144, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{160, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{160, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{160, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x224x128_16x16_16x16_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{160, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x160x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{176, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{176, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x224x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{176, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x224x128_16x16_16x16_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{176, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x224x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{192, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x192x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{192, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x224x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{192, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x224x128_16x16_16x16_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{192, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x224x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{208, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{208, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{208, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{208, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{224, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{224, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{224, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{224, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x224x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{240, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{240, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{240, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x224x128_16x16_16x16_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{240, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x256x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 10240, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 8192, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 57344, 8192},                                                                                                       \
        a8w8_bpreshuffle_256x256x224x128_16x16_16x16_8x32x1_8x32x1_1x64x1x4_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
       {{256, 8192, 28672},                                                                                                       \
        a8w8_bpreshuffle_256x256x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8x8x1_2x1_intrawave_v3<DTYPE, ETYPE>},                       \
   }

#endif // USE_ROCM
