#pragma once

#ifndef KCC_UTILS_CONSTS_H
#define KCC_UTILS_CONSTS_H

namespace kcc {
// keep types and constants in kcc namespace to avoid "using utils::TYPE"

enum {
    FMAP_TENSOR_RANK = 4,
    FILTER_TENSOR_RANK = 4,

    FmapIndex_N = 0,
    FmapIndex_C = 1,
    FmapIndex_H = 2,
    FmapIndex_W = 3,

    FilterIndex_M = 0,
    FilterIndex_C = 1,
    FilterIndex_R = 2,
    FilterIndex_S = 3,

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


constexpr const char* const TypeStr_Input     = "Input";
constexpr const char* const TypeStr_Conv      = "Conv";
constexpr const char* const TypeStr_Relu      = "Relu";
constexpr const char* const TypeStr_Tanh      = "Tanh";

constexpr const char* const TypeStr_MaxPool   = "MaxPool";
constexpr const char* const TypeStr_SoftMax   = "SoftMax";
constexpr const char* const TypeStr_BatchNorm = "BNorm";

constexpr const char* const Key_Layers        = "layers";
constexpr const char* const Key_NetName       = "net_name";
constexpr const char* const Key_DataType      = "data_type";

constexpr static const char* Key_LayerType        = "layer_type";
constexpr static const char* Key_LayerName        = "layer_name";
constexpr static const char* Key_PrevLayers       = "previous_layers";
constexpr static const char* Key_OfmapShape       = "ofmap_shape";

constexpr static const char* Key_OfmapFormat      = "ofmap_format";   // input,conv
constexpr static const char* Key_RefFile          = "ref_file";       // input

constexpr static const char* Key_KernelFile       = "kernel_file";    // input, conv
constexpr static const char* Key_KernelFormat     = "kernel_format";  // conv, pool
constexpr static const char* Key_KernelShape      = "kernel_shape";   // conv,pool
constexpr static const char* Key_Stride           = "stride";         // conv,pool
constexpr static const char* Key_Padding          = "padding";        // conv,pool
constexpr static const char* Key_Batching         = "batching";       //

} // namespace kcc

#endif
