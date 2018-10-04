#pragma once

#ifndef KCC_WAVE_CONSTS_H
#define KCC_WAVE_CONSTS_H

namespace kcc {
namespace wave {

constexpr const char* const WaveOpTypeStr_SBAtomLoad    = "SBAtomLoad";
constexpr const char* const WaveOpTypeStr_SBAtomSave    = "SBAtomSave";
constexpr const char* const WaveOpTypeStr_MatMul        = "MatMul";
constexpr const char* const WaveOpTypeStr_Pool          = "Pool";
constexpr const char* const WaveOpTypeStr_Activation    = "Activation";
constexpr const char* const WaveOpTypeStr_ResAdd        = "ResAdd";
constexpr const char* const WaveOpTypeStr_Multiply      = "Multiply";
constexpr const char* const WaveOpTypeStr_Barrier       = "Barrier";
constexpr const char* const WaveOpTypeStr_Nop           = "Nop";
constexpr const char* const WaveOpTypeStr_ScaleAdd      = "ScaleAdd";
constexpr const char* const WaveOpTypeStr_ClipByValue   = "ClipByValue";
constexpr const char* const WaveOpTypeStr_Maximum       = "Maximum";
constexpr const char* const WaveOpTypeStr_Minimum       = "Minimum";

}}

#endif

