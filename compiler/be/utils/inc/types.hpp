#pragma once

#ifndef KCC_UTILS_TYPES_H
#define KCC_UTILS_TYPES_H

#include "consts.hpp"

namespace kcc {
// keep types and constants in kcc namespace to avoid "using utils::TYPE"

typedef signed char     kcc_int8;
typedef short           kcc_int16;
typedef int             kcc_int32;
typedef long            kcc_int64;
typedef unsigned char   kcc_uint8;
typedef unsigned short  kcc_uint16;
typedef unsigned int    kcc_uint32;
typedef unsigned long   kcc_uint64;
typedef float           kcc_float32;
typedef double          kcc_float64;

static_assert(sizeof(kcc_int8)  == 1, "sizeof(int8) != 1");
static_assert(sizeof(kcc_int16) == 2, "sizeof(int16) != 2");
static_assert(sizeof(kcc_int32) == 4, "sizeof(int32) != 4");
static_assert(sizeof(kcc_int64) == 8, "sizeof(int64) != 8");
static_assert(sizeof(kcc_uint8)  == 1, "sizeof(uint8) != 1");
static_assert(sizeof(kcc_uint16) == 2, "sizeof(uint16) != 2");
static_assert(sizeof(kcc_uint32) == 4, "sizeof(uint32) != 4");
static_assert(sizeof(kcc_uint64) == 8, "sizeof(uint64) != 8");




typedef kcc_int32 LayerId;
const LayerId LayerId_Null = -1;


typedef kcc_int64 StateBufferAddress;
enum : StateBufferAddress {
    StateBufferAddress_Invalid = -1L
};

typedef kcc_int32 OfmapShapeType[FMAP_TENSOR_RANK];
typedef kcc_int32 KernelShapeType[FILTER_TENSOR_RANK];
typedef kcc_int32 StrideType[FMAP_TENSOR_RANK];
typedef kcc_int32 PaddingType[FMAP_TENSOR_RANK][2];
typedef kcc_int32 BatchingType[FMAP_TENSOR_RANK];

}

#endif

