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


constexpr const char* const LayerTypeStr_Input          = "Input";
constexpr const char* const LayerTypeStr_Const          = "Const";
constexpr const char* const LayerTypeStr_ResAdd         = "ResAdd";
constexpr const char* const LayerTypeStr_Multiply       = "Multiply";
constexpr const char* const LayerTypeStr_BiasAdd        = "BiasAdd";
constexpr const char* const LayerTypeStr_Conv           = "Conv";
constexpr const char* const LayerTypeStr_Reshape        = "Reshape";
constexpr const char* const LayerTypeStr_Matmul         = "MatMul";
constexpr const char* const LayerTypeStr_Relu           = "Relu";
constexpr const char* const LayerTypeStr_Tanh           = "Tanh";
constexpr const char* const LayerTypeStr_Sigmoid        = "Sigmoid";
constexpr const char* const LayerTypeStr_MaxPool        = "MaxPool";
constexpr const char* const LayerTypeStr_AvgPool        = "AvgPool";
constexpr const char* const LayerTypeStr_SoftMax        = "SoftMax";
constexpr const char* const LayerTypeStr_BatchNorm      = "BNorm";
constexpr const char* const LayerTypeStr_StridedSlice   = "StridedSlice";
constexpr const char* const LayerTypeStr_Unstack        = "Unstack";
constexpr const char* const LayerTypeStr_Concat         = "Concat";
constexpr const char* const LayerTypeStr_ConvTranspose  = "ConvTranspose";
constexpr const char* const LayerTypeStr_ClipByValue    = "ClipByValue";
constexpr const char* const LayerTypeStr_Squeeze        = "Squeeze";
constexpr const char* const LayerTypeStr_ExpandDims     = "ExpandDims";
constexpr const char* const LayerTypeStr_Slice          = "Slice";
constexpr const char* const LayerTypeStr_Minimum        = "Minimum";
constexpr const char* const LayerTypeStr_Pad            = "Pad";
constexpr const char* const LayerTypeStr_Softplus       = "Softplus";
constexpr const char* const LayerTypeStr_Transpose      = "Transpose";
constexpr const char* const LayerTypeStr_SpaceToBatchND = "SpaceToBatchND";
constexpr const char* const LayerTypeStr_BatchToSpaceND = "BatchToSpaceND";

constexpr const char* const WaveOpTypeStr_SBAtomLoad    = "SBAtomLoad";
constexpr const char* const WaveOpTypeStr_SBAtomSave    = "SBAtomSave";
constexpr const char* const WaveOpTypeStr_MatMul        = "MatMul";
constexpr const char* const WaveOpTypeStr_Pool          = "Pool";
constexpr const char* const WaveOpTypeStr_Activation    = "Activation";
constexpr const char* const WaveOpTypeStr_ResAdd        = "ResAdd";
constexpr const char* const WaveOpTypeStr_Barrier       = "Barrier";
constexpr const char* const WaveOpTypeStr_Nop           = "Nop";
constexpr const char* const WaveOpTypeStr_ScaleAdd      = "ScaleAdd";


constexpr const char* const EngineIdStr_PeArray         = "PeArrayEng";
constexpr const char* const EngineIdStr_Pool            = "PoolEng";
constexpr const char* const EngineIdStr_Activation      = "ActivationEng";
constexpr const char* const EngineIdStr_StreamProc      = "StreamProcEng";
constexpr const char* const EngineIdStr_Dma             = "DmaEng";

} // namespace kcc

#endif
