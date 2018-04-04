#pragma once

#ifndef KCC_UTILS_TYPES_H
#define KCC_UTILS_TYPES_H

#include <string>

#include "utils/inc/consts.hpp"

namespace kcc {

//**********************************************************************
enum IntTag {
    SbPartitionAddressTag,
    TpbAddressTag,
    TongaAddressTag,
};

//**********************************************************************
template <typename IntType, IntTag tag>
class Integer {
private:
    using Type = IntType;
public:
    Type gValue() const {
        return m_Value;
    }

private:
    Type m_Value;
};


//**********************************************************************
using kcc_int8    = signed char;
using kcc_int16   = short;
using kcc_int32   = int;
using kcc_int64   = long;
using kcc_uint8   = unsigned char;
using kcc_uint16  = unsigned short;
using kcc_uint32  = unsigned int;
using kcc_uint64  = unsigned long;
using kcc_float32 = float;
using kcc_float64 = double;


static_assert(sizeof(kcc_int8)  == 1, "sizeof(int8) != 1");
static_assert(sizeof(kcc_int16) == 2, "sizeof(int16) != 2");
static_assert(sizeof(kcc_int32) == 4, "sizeof(int32) != 4");
static_assert(sizeof(kcc_int64) == 8, "sizeof(int64) != 8");
static_assert(sizeof(kcc_uint8)  == 1, "sizeof(uint8) != 1");
static_assert(sizeof(kcc_uint16) == 2, "sizeof(uint16) != 2");
static_assert(sizeof(kcc_uint32) == 4, "sizeof(uint32) != 4");
static_assert(sizeof(kcc_uint64) == 8, "sizeof(uint64) != 8");

//**********************************************************************
using SbPartitionAddress    = Integer<kcc_int16, SbPartitionAddressTag>;
using TpbAddress            = Integer<kcc_int32, TpbAddressTag>;
using TongaAddress          = Integer<kcc_int64, TongaAddressTag>;


enum class EngineId {
    None,
    PeArray,
    Activation,
    Pooling,
    StreamProc,
    DmaEng,
    AnyEng, // when an instruction can be execed on several engines: WAIT,SET,WRITE
};
enum { NumberRealEngines = 5 };  // PeArray, ..., DmaEng

//**********************************************************************
enum class WaveOpType {
    Load,
    Save,
    Pool,
    MatMul,
    Activation,
    ResAdd,
    Barrier,
    Nop,
    ScaleAdd,
};

//**********************************************************************
enum class ActivationFunc {
    Invalid,
    Identity,
    Relu,
    LeakyRelu,
    PRelu,
    Sigmoid,
    Tanh,
    Exp,
};

//**********************************************************************
enum class PoolType {
    None,
    Max,
    Avg
};


//**********************************************************************
using LayerId = kcc_int32;
const LayerId LayerId_Null = -1;


using StateBufferAddress = kcc_int64;
enum : StateBufferAddress {
    StateBufferAddress_Invalid = -1L
};

using OfmapShapeType  = kcc_int32[FMAP_TENSOR_RANK];
using KernelShapeType = kcc_int32[FILTER_TENSOR_RANK];
using StrideType      = kcc_int32[FMAP_TENSOR_RANK];
using PaddingType     = kcc_int32[FMAP_TENSOR_RANK][2];
using BatchingType    = kcc_int32[FMAP_TENSOR_RANK];




namespace utils {

const std::string& poolType2Str(PoolType poolType);
PoolType           poolTypeStr2Id(const std::string&);
const std::string& engineId2Str(EngineId engId);
EngineId engineId2Str(const std::string& str);

constexpr kcc_int64 power2(kcc_int64 b) {
    return 1 << (b);


class MemAccessPatternXY {
    kcc_int32       m_XStep             = -1;
    kcc_int32       m_XNum              = -1;
    kcc_int32       m_YStep             = -1;
    kcc_int32       m_YNum              = -1;
};

class MemAccessPatternXYZ {
    kcc_int32       m_XStep             = -1;
    kcc_int32       m_XNum              = -1;
    kcc_int32       m_YStep             = -1;
    kcc_int32       m_YNum              = -1;
    kcc_int32       m_ZStep             = -1;
    kcc_int32       m_ZNum              = -1;
};

class MemAccessPatternXYZW {
    kcc_int32       m_XStep             = -1;
    kcc_int32       m_XNum              = -1;
    kcc_int32       m_YStep             = -1;
    kcc_int32       m_YNum              = -1;
    kcc_int32       m_ZStep             = -1;
    kcc_int32       m_ZNum              = -1;
    kcc_int32       m_WStep             = -1;
    kcc_int32       m_WNum              = -1;
};

}

}

}

#endif

