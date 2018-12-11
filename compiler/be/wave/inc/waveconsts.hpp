#pragma once

#ifndef KCC_WAVE_CONSTS_H
#define KCC_WAVE_CONSTS_H

namespace kcc {
namespace wave {

namespace WaveOpTypeStr {
constexpr const char* const SBAtomLoad    = "SBAtomLoad";
constexpr const char* const SBAtomSave    = "SBAtomSave";
constexpr const char* const TpbCopy       = "TpbCopy";
constexpr const char* const MatMul        = "MatMul";
constexpr const char* const Pool          = "Pool";
constexpr const char* const Reciprocal    = "Reciprocal";
constexpr const char* const RegLoad       = "RegLoad";
constexpr const char* const RegStore      = "RegStore";
constexpr const char* const Activation    = "Activation";
constexpr const char* const Barrier       = "Barrier";

constexpr const char* const ResAdd        = "ResAdd";
constexpr const char* const Nop           = "Nop";
constexpr const char* const ScaleAdd      = "ScaleAdd";
constexpr const char* const Multiply      = "Multiply";
constexpr const char* const Add           = "Add";
constexpr const char* const Sub           = "Sub";
constexpr const char* const ClipByValue   = "ClipByValue";
constexpr static const char* TensorTensor = "TensorTensor";
constexpr static const char* TensorScalar = "TensorScalar";
constexpr static const char* TensorScalarPtr = "TensorScalarPtr";
constexpr const char* const Maximum       = "Maximum";
constexpr const char* const Minimum       = "Minimum";

constexpr const char* const MaxPool       = "MaxPool";
constexpr const char* const AvgPool       = "AvgPool";
} // namespace WaveOpTypeStr

}}

#endif

