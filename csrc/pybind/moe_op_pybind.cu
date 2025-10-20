/* SPDX-License-Identifier: MIT
   Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
*/
#include "moe_op.h"
#include "rocm_ops.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // REMOVED: AITER_ENUM_PYBIND - enums already registered in module_aiter_enum
    // This fixes "QuantType already registered" error in multi-GPU
    MOE_OP_PYBIND;
}