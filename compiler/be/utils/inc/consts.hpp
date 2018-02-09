#pragma once

#ifndef KCC_UTILS_CONSTS_H
#define KCC_UTILS_CONSTS_H

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
constexpr const char* const LayerTypeStr_BiasAdd        = "BiasAdd";
constexpr const char* const LayerTypeStr_Conv           = "Conv";
constexpr const char* const LayerTypeStr_Relu           = "Relu";
constexpr const char* const LayerTypeStr_Tanh           = "Tanh";

constexpr const char* const LayerTypeStr_MaxPool        = "MaxPool";
constexpr const char* const LayerTypeStr_AvgPool        = "AvgPool";
constexpr const char* const LayerTypeStr_SoftMax        = "SoftMax";
constexpr const char* const LayerTypeStr_BatchNorm      = "BNorm";

constexpr const char* const WaveOpTypeStr_SBAtomFile    = "SBAtomFile";
constexpr const char* const WaveOpTypeStr_SBAtomSave    = "SBAtomSave";
constexpr const char* const WaveOpTypeStr_MatMul        = "MatMul";

constexpr const char* const NetKey_Layers               = "layers";
constexpr const char* const NetKey_WaveOps              = "waveops";
constexpr const char* const NetKey_NetName              = "net_name";
constexpr const char* const NetKey_DataType             = "data_type";

// Layer keys

constexpr static const char* LayerKey_LayerType         = "layer_type";
constexpr static const char* LayerKey_LayerName         = "layer_name";
constexpr static const char* LayerKey_PrevLayers        = "previous_layers";
constexpr static const char* LayerKey_OfmapShape        = "ofmap_shape";
    
constexpr static const char* LayerKey_OfmapFormat       = "ofmap_format";
constexpr static const char* LayerKey_RefFile           = "ref_file";

constexpr static const char* LayerKey_KernelFile        = "kernel_file";
constexpr static const char* LayerKey_KernelFormat      = "kernel_format";
constexpr static const char* LayerKey_KernelShape       = "kernel_shape";
constexpr static const char* LayerKey_Stride            = "stride";
constexpr static const char* LayerKey_Padding           = "padding";
constexpr static const char* LayerKey_Batching          = "batching";



// common to all WaveOps
constexpr static const char* WaveOpKey_WaveOpType           = "waveop_type";
constexpr static const char* WaveOpKey_WaveOpName           = "waveop_name";
constexpr static const char* WaveOpKey_LayerName            = "layer_name";
constexpr static const char* WaveOpKey_PreviousWaveOps      = "previous_waveops";

// MatMul
constexpr static const char* WaveOpKey_BatchingInWave       = "batching_in_wave";
constexpr static const char* WaveOpKey_IfmapCount           = "ifmap_count";
constexpr static const char* WaveOpKey_IfmapTileHeight      = "ifmap_tile_height";
constexpr static const char* WaveOpKey_IfmapTileWidth       = "ifmap_tile_width";
constexpr static const char* WaveOpKey_IfmapsAtomId         = "ifmaps_atom_id";
constexpr static const char* WaveOpKey_IfmapsOffsetInAtom   = "ifmaps_offset_in_atom";
// layer name
constexpr static const char* WaveOpKey_OfmapCount           = "ofmap_count";
constexpr static const char* WaveOpKey_OfmapTileHeight      = "ofmap_tile_height";
constexpr static const char* WaveOpKey_OfmapTileWidth       = "ofmap_tile_width";
// previous waveops
constexpr static const char* WaveOpKey_PsumBankId           = "psum_bank_id";
constexpr static const char* WaveOpKey_PsumBankOffset       = "psum_bank_offset";
constexpr static const char* WaveOpKey_Start                = "start";
constexpr static const char* WaveOpKey_WaveId               = "wave_id";
constexpr static const char* WaveOpKey_WaveIdFormat         = "wave_id_format";
// waveop name
// waveop type
constexpr static const char* WaveOpKey_WeightsAtomId        = "weights_atom_id";
constexpr static const char* WaveOpKey_WeightsOffsetInAtom  = "weights_offset_in_atom";

// SBAtom common
constexpr static const char* WaveOpKey_AtomId               = "atom_id";
constexpr static const char* WaveOpKey_BatchFoldIdx         = "batch_fold_idx";
constexpr static const char* WaveOpKey_Length               = "length";
constexpr static const char* WaveOpKey_OffsetInFile         = "offset_in_file";
constexpr static const char* WaveOpKey_RefFile              = "ref_file";

// SBAtomFile
constexpr static const char* WaveOpKey_IfmapsFoldIdx        = "ifmaps_fold_idx";
constexpr static const char* WaveOpKey_IfmapsReplicate      = "ifmaps_replicate";

// SBAtomSave
constexpr static const char* WaveOpKey_OfmapsFoldIdx = "ofmaps_fold_idx";

} // namespace kcc

#endif
