#pragma once

#ifndef KCC_UTILS_CONSTS_H
#define KCC_UTILS_CONSTS_H

#define KCC_CONCAT2(x,y) x##y
#define KCC_CONCAT(x,y) KCC_CONCAT2(x,y)
#define KCC_STR(x) #x

namespace kcc {

enum {
    FMAP_TENSOR_RANK = 4,
    FILTER_TENSOR_RANK = 4,

    FmapIndex_N = 0,
    FmapIndex_C = 1,
    FmapIndex_H = 2,
    FmapIndex_W = 3,

    FilterIndex_M = 3,
    FilterIndex_C = 0,
    FilterIndex_R = 1,
    FilterIndex_S = 2,

    PaddingIndex_Top    = 0,
    PaddingIndex_Bottom = 1,
    PaddingIndex_Left   = 2,
    PaddingIndex_Right  = 3,

    StrideIndex_TopBottom = 0,
    StrideIndex_LeftRight = 1,

    DilateIndex_TopBottom = 0,
    DilateIndex_LeftRight = 1,
};

static_assert(StrideIndex_TopBottom == DilateIndex_TopBottom, "Stride and Dilate TopBottom indices not same");

static_assert(StrideIndex_LeftRight == DilateIndex_LeftRight, "Stride and Dilate LeftRight indices not same");

const char* const SCHED_MEM_FORMAT = "%-24s %10s %6s  %8s  %8s  %8s  %8s";

constexpr const char* const EngineIdStr_PeArray         = "PeArrayEng";
constexpr const char* const EngineIdStr_Pool            = "PoolEng";
constexpr const char* const EngineIdStr_Activation      = "ActivationEng";
constexpr const char* const EngineIdStr_StreamProc      = "StreamProcEng";
constexpr const char* const EngineIdStr_Dma             = "DmaEng";

} // namespace kcc

#endif
