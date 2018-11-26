#pragma once

#ifndef KCC_LAYERS_CONSTS_H
#define KCC_LAYERS_CONSTS_H

namespace kcc {
namespace layers {

constexpr const char* const LayerTypeStr_Input          = "Input";
constexpr const char* const LayerTypeStr_Const          = "Const";
constexpr const char* const LayerTypeStr_ResAdd         = "ResAdd";
constexpr const char* const LayerTypeStr_Add            = "Add";
constexpr const char* const LayerTypeStr_Sub            = "Sub";
constexpr const char* const LayerTypeStr_Multiply       = "Multiply";
constexpr const char* const LayerTypeStr_BiasAdd        = "BiasAdd";
constexpr const char* const LayerTypeStr_Conv           = "Conv";
constexpr const char* const LayerTypeStr_Reshape        = "Reshape";
constexpr const char* const LayerTypeStr_Matmul         = "MatMul";
constexpr const char* const LayerTypeStr_Relu           = "Relu";
constexpr const char* const LayerTypeStr_Tanh           = "Tanh";
constexpr const char* const LayerTypeStr_Sigmoid        = "Sigmoid";
constexpr const char* const LayerTypeStr_Softplus       = "Softplus";
constexpr const char* const LayerTypeStr_Sqrt           = "Sqrt";
constexpr const char* const LayerTypeStr_MaxPool        = "MaxPool";
constexpr const char* const LayerTypeStr_AvgPool        = "AvgPool";
constexpr const char* const LayerTypeStr_SoftMax        = "SoftMax";
constexpr const char* const LayerTypeStr_BatchNorm      = "BNorm";
constexpr const char* const LayerTypeStr_StridedSlice   = "StridedSlice";
constexpr const char* const LayerTypeStr_Unstack        = "Unstack";
constexpr const char* const LayerTypeStr_Concat         = "Concat";
constexpr const char* const LayerTypeStr_ConvTranspose  = "ConvTranspose";
constexpr const char* const LayerTypeStr_ClipByValue    = "ClipByValue";
constexpr const char* const LayerTypeStr_Split          = "Split";
constexpr const char* const LayerTypeStr_Squeeze        = "Squeeze";
constexpr const char* const LayerTypeStr_ExpandDims     = "ExpandDims";
constexpr const char* const LayerTypeStr_Slice          = "Slice";
constexpr const char* const LayerTypeStr_Minimum        = "Minimum";
constexpr const char* const LayerTypeStr_Maximum        = "Maximum";
constexpr const char* const LayerTypeStr_Pad            = "Pad";
constexpr const char* const LayerTypeStr_Transpose      = "Transpose";
constexpr const char* const LayerTypeStr_SpaceToBatchND = "SpaceToBatchND";
constexpr const char* const LayerTypeStr_BatchToSpaceND = "BatchToSpaceND";
constexpr const char* const LayerTypeStr_Dequantize     = "Dequantize";
constexpr const char* const LayerTypeStr_Quantize       = "QuantizeV2";
constexpr const char* const LayerTypeStr_QuantizedConv  = "QuantizedConv";

}}

#endif
