#pragma once

#ifndef KCC_UTILS_CONSTS_H
#define KCC_UTILS_CONSTS_H

namespace kcc {
namespace utils {

const char* const SCHED_MEM_FORMAT = "%s";

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

} // namespace utils
} // namespace kcc

#endif
