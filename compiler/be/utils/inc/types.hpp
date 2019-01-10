#pragma once

#ifndef KCC_UTILS_TYPES_H
#define KCC_UTILS_TYPES_H

#include <string>
#include <array>

#include "aws_tonga_isa_common.h"
#include "aws_tonga_isa_tpb_common.h"

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

static_assert(sizeof(kcc_float32) == 4, "sizeof(float32) != 4");
static_assert(sizeof(kcc_float64) == 8, "sizeof(float64) != 8");

//**********************************************************************
//using SbPartitionAddress    = Integer<kcc_int16, SbPartitionAddressTag>;
//using TpbAddress            = Integer<kcc_int32, TpbAddressTag>;
//using TongaAddress          = Integer<kcc_int64, TongaAddressTag>;

#if true
using TpbAddress            = tpb_addr;
using SbPartitionAddress    = TpbAddress;
using TongaAddress          = tonga_addr;
#else
using TpbAddress = Integer<tpb_addr, TpbAddressTag>;
using SbPartitionAddress    = Integer<tpb_addr, SbPartitionAddressTag>;
using TongaAddress = Integer<tonga_addr, TongaAddressTag>;
#endif


enum class BinFileType {
    SimAngel,
    RuntimeKelf
};

enum class EngineId { // order Pooling,Activation,PeArray by 'busyness'
    None,
    Pooling,
    Activation,
    PeArray,
    StreamProc,
    AngelEng,
    AnyEng, // when an instruction can be execed on several engines: WAIT,SET,WRITE
};


//**********************************************************************
enum class WaveOpType {
    Load,
    Save,
    Pool,
    Reciprocal,
    RegLoad,
    RegStore,
    RegShuffle,
    MatMul,
    Activation,
    TensorTensor,
    TensorScalar,
    TensorScalarPtr,
    Nop,
    ScaleAdd,
    ClipByValue,
    TpbCopy,

    // Must be last
    Count
};

#undef CONST
#define CONST(X) constexpr static const char* KCC_CONCAT(TensorAluTypeStr_,X) = KCC_STR(X);
enum class TensorAluOpType {
    Bypass          = TONGA_ISA_TPB_ALU_OP_BYPASS,
    BwNot           = TONGA_ISA_TPB_ALU_OP_BITWISE_NOT,
    ArithShiftLeft  = TONGA_ISA_TPB_ALU_OP_ARITH_SHIFT_LEFT,
    ArithShiftRight = TONGA_ISA_TPB_ALU_OP_ARITH_SHIFT_RIGHT,
    Add             = TONGA_ISA_TPB_ALU_OP_ADD,
    Sub             = TONGA_ISA_TPB_ALU_OP_SUBTRACT,
    Mult            = TONGA_ISA_TPB_ALU_OP_MULT,
    Div             = TONGA_ISA_TPB_ALU_OP_DIVIDE,
    Max             = TONGA_ISA_TPB_ALU_OP_MAX,
    Min             = TONGA_ISA_TPB_ALU_OP_MIN,
    BwAnd           = TONGA_ISA_TPB_ALU_OP_BITWISE_AND,
    BwOr            = TONGA_ISA_TPB_ALU_OP_BITWISE_OR,
    BwXor           = TONGA_ISA_TPB_ALU_OP_BITWISE_XOR,
    LogAnd          = TONGA_ISA_TPB_ALU_OP_LOGICAL_AND,
    LogOr           = TONGA_ISA_TPB_ALU_OP_LOGICAL_OR,
    LogXor          = TONGA_ISA_TPB_ALU_OP_LOGICAL_XOR,
    LogShiftLeft    = TONGA_ISA_TPB_ALU_OP_LOGICAL_SHIFT_LEFT,
    LogShiftRight   = TONGA_ISA_TPB_ALU_OP_LOGICAL_SHIFT_RIGHT,
    Equal           = TONGA_ISA_TPB_ALU_OP_IS_EQUAL,
    Gt              = TONGA_ISA_TPB_ALU_OP_IS_GT,
    Ge              = TONGA_ISA_TPB_ALU_OP_IS_GE,
    Lt              = TONGA_ISA_TPB_ALU_OP_IS_LT,
    Le              = TONGA_ISA_TPB_ALU_OP_IS_LE,
   Number,
};

//**********************************************************************
/*
#undef CONST
#define CONST(X) constexpr static const char* TensorAluTypeStr_::X = KCC_STR(X);
*/

namespace TensorAluTypeStr {
constexpr static const char* Bypass             = "bypass";
constexpr static const char* BwNot              = "bitwise_not";
constexpr static const char* ArithShiftLeft     = "arith_shift_left";
constexpr static const char* ArithShiftRight    = "arith_shift_right";
constexpr static const char* Add                = "add";
constexpr static const char* Sub                = "subtract";
constexpr static const char* Mult               = "mult";
constexpr static const char* Div                = "divide";
constexpr static const char* Max                = "max";
constexpr static const char* Min                = "min";
constexpr static const char* BwAnd              = "bitwise_and";
constexpr static const char* BwOr               = "bitwise_or";
constexpr static const char* BwXor              = "bitwise_xor";
constexpr static const char* LogAnd             = "logical_and";
constexpr static const char* LogOr              = "logical_or";
constexpr static const char* LogXor             = "logical_xor";
constexpr static const char* LogShiftLeft       = "logical_shift_left";
constexpr static const char* LogShiftRight      = "logical_shift_right";
constexpr static const char* Equal              = "is_equal";
constexpr static const char* Gt                 = "is_gt";
constexpr static const char* Ge                 = "is_ge";
constexpr static const char* Lt                 = "is_lt";
constexpr static const char* Le                 = "is_le";
}



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
    Sqrt,
    Softplus,
};


//**********************************************************************
enum class PoolType {
    None,
    Max,
    Avg
};
namespace PoolTypeStr {
constexpr const char* const MaxPool       = "MaxPool";
constexpr const char* const AvgPool       = "AvgPool";
}


//**********************************************************************


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
int cmpEngineId(EngineId lhs, EngineId rhs);

const std::string& ActivationFunc2Str(ActivationFunc);



constexpr kcc_int64 power2(kcc_int64 b) {
    return 1 << (b);
}

enum class Dims {
    X,
    XY,
    XYZ,
    XYZW
};

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


class TensorParams {
public:
    enum { NUM_DIMS = 5 };
    using ShapeType = std::array<kcc_int32, NUM_DIMS>;
public:
    TensorParams(const ShapeType& shape, const char* format)
    {
        m_Shape = shape;
        for (int i = 0; i < NUM_DIMS; ++i) {
            m_Format[i] = format[i];
        }
        m_Format[NUM_DIMS] = '\0';
    }
    TensorParams() = delete;

    const char* gFormat() const {
        return m_Format;
    }
    kcc_int32 operator[] (kcc_int32 n) const {
        return m_Shape[n];
    }

    kcc_int32 size() const {
        return m_Shape.size();
    }

    kcc_int64 gSize() const {
        kcc_int64 sz = 1;
        for (auto n : m_Shape) {
            sz *= n;
        }
        return sz;
    }

    class iterator;

    inline iterator begin() const;
    inline iterator end() const;

private:
    ShapeType m_Shape;
    char m_Format[NUM_DIMS+1];
};

class TensorParams::iterator {
public:
    iterator(const TensorParams& params, kcc_int32 idx)
        : m_Params(params)
        , m_Idx(idx)
    {
    }

    bool operator!= (const iterator& rhs) const {
        return rhs.m_Idx != m_Idx;
    }

    kcc_int32 operator* () const {
        return m_Params[m_Idx];
    }

    void operator++ () {
        ++m_Idx;
    }

private:
    const TensorParams& m_Params;
    kcc_int32 m_Idx;
};


inline auto
TensorParams::begin() const -> iterator
{
    return iterator(*this, 0);
}

inline auto
TensorParams::end() const -> iterator
{
    return iterator(*this, NUM_DIMS);
}

extern TensorAluOpType gAluOpType(const char* tensorOp);
extern const char* gAluOpTypeStr(TensorAluOpType opType);

enum class PEPerfOptType {
    None            = TONGA_ISA_TPB_PE_PERF_OPT_NONE,
    DoubleRow       = TONGA_ISA_TPB_PE_PERF_OPT_DOUBLE_ROW,
    DoubleColumn    = TONGA_ISA_TPB_PE_PERF_OPT_DOUBLE_COLUMN,
    DoublePixel     = TONGA_ISA_TPB_PE_PERF_OPT_DOUBLE_PIXEL,
};

namespace PEPerfOptTypeStr {
    constexpr static const char* None           = "none";
    constexpr static const char* DoubleRow      = "double_row";
    constexpr static const char* DoubleColumn   = "double_column";
    constexpr static const char* DoublePixel    = "double_pixel";
}

extern PEPerfOptType gPEPerfOptType(const char*);
extern const char* gPEPerfOptTypeStr(PEPerfOptType);

}} // namespace utils, kcc

#endif

