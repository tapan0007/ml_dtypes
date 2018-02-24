#pragma once

#ifndef KCC_SERIALIZE_SERWAVEOP_H
#define KCC_SERIALIZE_SERWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>

#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>




#include "utils/inc/debug.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "utils/inc/datatype.hpp"

// Must include matmulwaveop.hpp for WaveId
#include "wave/inc/matmulwaveop.hpp"


namespace kcc {
using  namespace utils;

namespace serialize {


// common to all WaveOps
constexpr static const char* WaveOpKey_WaveOpType           = "waveop_type";
constexpr static const char* WaveOpKey_WaveOpName           = "waveop_name";
constexpr static const char* WaveOpKey_LayerName            = "layer_name";
constexpr static const char* WaveOpKey_PreviousWaveOps      = "previous_waveops";


// MatMul
constexpr static const char* WaveOpKey_BatchingInWave       = "batching_in_wave";
constexpr static const char* WaveOpKey_FmapXNum             = "fmap_x_num";
constexpr static const char* WaveOpKey_FmapXStep            = "fmap_x_step";
constexpr static const char* WaveOpKey_FmapYNum             = "fmap_y_num";
constexpr static const char* WaveOpKey_FmapYStep            = "fmap_y_step";
constexpr static const char* WaveOpKey_FmapZNum             = "fmap_z_num";
constexpr static const char* WaveOpKey_FmapZStepAtoms       = "fmap_z_step_atoms";
constexpr static const char* WaveOpKey_IfmapCount           = "ifmap_count";
constexpr static const char* WaveOpKey_IfmapTileHeight      = "ifmap_tile_height";
constexpr static const char* WaveOpKey_IfmapTileWidth       = "ifmap_tile_width";
constexpr static const char* WaveOpKey_IfmapsAtomId         = "ifmaps_atom_id";
constexpr static const char* WaveOpKey_IfmapsAtomSize       = "ifmaps_atom_size";
constexpr static const char* WaveOpKey_DataType             = "data_type";
constexpr static const char* WaveOpKey_IfmapsOffsetInAtom   = "ifmaps_offset_in_atom";
// layer name
constexpr static const char* WaveOpKey_NumColumnPartitions  = "num_column_partitions";
constexpr static const char* WaveOpKey_NumRowPartitions     = "num_row_partitions";
constexpr static const char* WaveOpKey_OfmapCount           = "ofmap_count";
constexpr static const char* WaveOpKey_OfmapTileHeight      = "ofmap_tile_height";
constexpr static const char* WaveOpKey_OfmapTileWidth       = "ofmap_tile_width";
// previous waveops
constexpr static const char* WaveOpKey_PsumBankId           = "psum_bank_id";
constexpr static const char* WaveOpKey_PsumBankOffset       = "psum_bank_offset";
constexpr static const char* WaveOpKey_PsumXNum             = "psum_x_num";
constexpr static const char* WaveOpKey_PsumXStep             = "psum_x_step";
constexpr static const char* WaveOpKey_PsumYNum             = "psum_y_num";
constexpr static const char* WaveOpKey_PsumYStep             = "psum_y_step";
constexpr static const char* WaveOpKey_StartTensorCalc      = "start_tensor_calc";
constexpr static const char* WaveOpKey_StopTensorCalc      = "stop_tensor_calc";
constexpr static const char* WaveOpKey_StrideX               = "stride_x";
constexpr static const char* WaveOpKey_StrideY               = "stride_y";
constexpr static const char* WaveOpKey_WaveId               = "wave_id";
constexpr static const char* WaveOpKey_WaveIdFormat         = "wave_id_format";
// waveop name
// waveop type
constexpr static const char* WaveOpKey_WeightsAtomId        = "weights_atom_id";
constexpr static const char* WaveOpKey_WeightsOffsetInAtom  = "weights_offset_in_atom";

// SBAtom common
constexpr static const char* WaveOpKey_AtomId               = "atom_id";
constexpr static const char* WaveOpKey_AtomSize             = "atom_size";
constexpr static const char* WaveOpKey_BatchFoldIdx         = "batch_fold_idx";
constexpr static const char* WaveOpKey_Length               = "length";
constexpr static const char* WaveOpKey_OffsetInFile         = "offset_in_file";
constexpr static const char* WaveOpKey_PartitionStepBytes   = "partition_step_bytes";
constexpr static const char* WaveOpKey_RefFile              = "ref_file";
constexpr static const char* WaveOpKey_RefFileFormat        = "ref_file_format";
constexpr static const char* WaveOpKey_RefFileShape         = "ref_file_shape";

// SBAtomFile
constexpr static const char* WaveOpKey_IfmapsFoldIdx        = "ifmaps_fold_idx";
constexpr static const char* WaveOpKey_IfmapsReplicate      = "ifmaps_replicate";

// SBAtomSave
constexpr static const char* WaveOpKey_OfmapsFoldIdx = "ofmaps_fold_idx";

// Pool
constexpr static const char* WaveOpKey_DstSbAtomId          = "dst_sb_atom_id";
constexpr static const char* WaveOpKey_DstSbOffsetInAtom    = "dst_sb_offset_in_atom";
constexpr static const char* WaveOpKey_DstXNum              = "dst_x_num";
constexpr static const char* WaveOpKey_DstXStep	            = "dst_x_step";
constexpr static const char* WaveOpKey_DstYNum	            = "dst_y_num";
constexpr static const char* WaveOpKey_DstYStep	            = "dst_y_step";
constexpr static const char* WaveOpKey_DstZNum	            = "dst_z_num";
constexpr static const char* WaveOpKey_DstZStep	            = "dst_z_step";
constexpr static const char* WaveOpKey_InDtype              = "in_dtype";
// "layername": "1conv/i1",
constexpr static const char* WaveOpKey_NumPartitions        = "num_partitions";
constexpr static const char* WaveOpKey_OutDtype             = "out_dtype";
constexpr static const char* WaveOpKey_PoolFrequency	    = "pool_frequency";
constexpr static const char* WaveOpKey_PoolFunc             = "pool_func";
// previouswaveops": [ 1conv/i1/MatMuln0m0h0w0c0r0s0" ]
constexpr static const char* WaveOpKey_SrcIsPsum            = "src_is_psum";
constexpr static const char* WaveOpKey_SrcPsumBankId        = "src_psum_bank_id";
constexpr static const char* WaveOpKey_SrcPsumBankOffset    = "src_psum_bank_offset";
constexpr static const char* WaveOpKey_SrcSbAtomId          = "src_sb_atom_id";
constexpr static const char* WaveOpKey_SrcSbOffsetInAtom    = "src_sb_offset_in_atom";
constexpr static const char* WaveOpKey_SrcWNum	            = "src_w_num";
constexpr static const char* WaveOpKey_SrcWStep	            = "src_w_step";
constexpr static const char* WaveOpKey_SrcXNum	            = "src_x_num";
constexpr static const char* WaveOpKey_SrcXStep	            = "src_x_step";
constexpr static const char* WaveOpKey_SrcYNum	            = "src_y_num";
constexpr static const char* WaveOpKey_SrcYStep	            = "src_y_step";
constexpr static const char* WaveOpKey_SrcZNum	            = "src_z_num";
constexpr static const char* WaveOpKey_SrcZStep	            = "src_z_step";
constexpr static const char* WaveOpKey_TileId               = "tile_id";
constexpr static const char* WaveOpKey_TileIdFormat         = "tile_id_format";
//waveopname": "1conv/i1/Pooln0m0h0w0",
//waveoptype": "Pool"

constexpr static const char* WaveOpKey_ActivationFunc              = "activation_func";
constexpr static const char* WaveOpKey_ActivationFunc_None         = "none"; /* until Jeff fixes none */
constexpr static const char* WaveOpKey_ActivationFunc_Identity     = "Identity";
constexpr static const char* WaveOpKey_ActivationFunc_Relu         = "Relu";
constexpr static const char* WaveOpKey_ActivationFunc_LeakyRelu    = "Lrelu";
constexpr static const char* WaveOpKey_ActivationFunc_Prelu        = "Prelu";
constexpr static const char* WaveOpKey_ActivationFunc_Sigmoid      = "Sigmoid";
constexpr static const char* WaveOpKey_ActivationFunc_Tanh         = "Tanh";
constexpr static const char* WaveOpKey_ActivationFunc_Exp          = "Exp";


constexpr static const char* WaveOpKey_BiasAddEn            = "bias_add_en";
constexpr static const char* WaveOpKey_BiasAtomId           = "bias_atom_id";
constexpr static const char* WaveOpKey_BiasOffsetInAtom     = "bias_offset_in_atom";
constexpr static const char* WaveOpKey_DstPsumBankId        = "dst_psum_bank_id";
//constexpr static const char* WaveOpKey_DstXNum              = "dst_x_num";
//constexpr static const char* WaveOpKey_DstXStep             = "dst_x_step";
//constexpr static const char* WaveOpKey_DstYNum              = "dst_y_num";
//constexpr static const char* WaveOpKey_DstYStep             = "dst_y_step";
//constexpr static const char* WaveOpKey_DstZNum              = "dst_z_num";
//constexpr static const char* WaveOpKey_DstZStep             = "dst_z_step";
//constexpr static const char* WaveOpKey_InDtype              = "in_dtype";
//constexpr static const char* WaveOpKey_NumPartitions        = "num_partitions";
//constexpr static const char* WaveOpKey_OutDtype             = "out_dtype";
//constexpr static const char* WaveOpKey_SrcPsumBankId        = "src_psum_bank_id";
//constexpr static const char* WaveOpKey_SrcXNum              = "src_x_num";
//constexpr static const char* WaveOpKey_SrcXStep             = "src_x_step";
//constexpr static const char* WaveOpKey_SrcYNum              = "src_y_num";
//constexpr static const char* WaveOpKey_SrcYStep             = "src_y_step";
//constexpr static const char* WaveOpKey_SrcZNum              = "src_z_num";
//constexpr static const char* WaveOpKey_SrcZStep             = "src_z_step";
//constexpr static const char* WaveOpKey_TileId               = "tile_id";
//constexpr static const char* WaveOpKey_TileIdFormat         = "tile_id_format";








class SerWaveOp {
public:
    SerWaveOp();


    SerWaveOp(const SerWaveOp&) = default;

public:

    template<typename Archive>
    void save(Archive & archive) const;

    template<typename Archive>
    void load(Archive & archive);


    void addPreviousWaveOp(const std::string& prevWaveOp) {
        m_PreviousWaveOps.push_back(prevWaveOp);
    }

    static ActivationFunc str2ActivationFunc(const std::string& s);
    static std::string activationType2Str(ActivationFunc);

private:
    void loadSbAtom(cereal::JSONInputArchive& archive);
    void loadPool(cereal::JSONInputArchive& archive);
    void loadMatMul(cereal::JSONInputArchive& archive);
    void loadActivation(cereal::JSONInputArchive& archive);

    void saveSbAtom(cereal::JSONOutputArchive& archive) const;
    void savePool(cereal::JSONOutputArchive& archive) const;
    void saveMatMul(cereal::JSONOutputArchive& archive) const;
    void saveActivation(cereal::JSONOutputArchive& archive) const;

protected:
    bool verify() const;

private:
    bool verifySbAtom() const;
    bool verifySbAtomFile() const;
    bool verifySbAtomSave() const;
    bool verifyMatMul() const;
    bool verifyPool() const;
    bool verifyActivation() const;

public:
    // common to all
    std::string                 m_WaveOpType        = "";
    std::string                 m_WaveOpName        = "";
    std::string                 m_LayerName         = "";
    std::vector<std::string>    m_PreviousWaveOps;

    // SBAtom
    kcc_int32                   m_AtomId            = -1;
    kcc_int32                   m_AtomSize          = -1;
    kcc_int32                   m_BatchFoldIdx      = -1;
    std::string                 m_DataType          = "";
    //layer name
    kcc_int64                   m_Length            = -1;
    kcc_int32                   m_OffsetInFile      = -1;
    kcc_int64                   m_PartitionStepBytes= -1;
    // previous waveops
    std::string                 m_RefFile           = "";
    std::string                 m_RefFileFormat     = "";
    std::vector<kcc_int32>      m_RefFileShape;
    // waveop name
    // waveop type

    // SBAtomFile
    kcc_int32                   m_IfmapCount       = -1;
    kcc_int32                   m_IfmapsFoldIdx     = -1;
    bool                        m_IfmapsReplicate   = false;

    // SBAtomSave
    kcc_int32                   m_OfmapCount       = -1;
    kcc_int32                   m_OfmapsFoldIdx     = -1;

    // MatMul
    enum {
        WaveIdFormatSize = 7,
    };
    kcc_int32                   m_BatchingInWave        = -1;
    kcc_int32                   m_FmapXNum              = -1;
    kcc_int32                   m_FmapXStep             = -1;
    kcc_int32                   m_FmapYNum              = -1;
    kcc_int32                   m_FmapYStep             = -1;
    kcc_int32                   m_FmapZNum              = -1;
    kcc_int32                   m_FmapZStepAtoms        = -1;
    //kcc_int32                   m_IfmapCount            = -1;
    kcc_int32                   m_IfmapTileHeight       = -1;
    kcc_int32                   m_IfmapTileWidth        = -1;
    kcc_int32                   m_IfmapsAtomId          = -1;
    kcc_int32                   m_IfmapsAtomSize        = -1;
    kcc_int32                   m_IfmapsOffsetInAtom    = -1;
    // layer name
    kcc_int32                   m_NumColumnPartitions   = -1;
    kcc_int32                   m_NumRowPartitions      = -1;
    //kcc_int32                   m_OfmapCount            = -1;
    kcc_int32                   m_OfmapTileHeight       = -1;
    kcc_int32                   m_OfmapTileWidth        = -1;
    // previous waveops
    kcc_int32                   m_PsumBankId            = -1;
    kcc_int32                   m_PsumBankOffset        = -1;
    kcc_int32                   m_PsumXNum              = -1;
    kcc_int32                   m_PsumXStep             = -1;
    kcc_int32                   m_PsumYNum              = -1;
    kcc_int32                   m_PsumYStep             = -1;
    bool                        m_StartTensorCalc       = true;
    bool                        m_StopTensorCalc        = true;
    kcc_int32                   m_StrideX               = -1;
    kcc_int32                   m_StrideY               = -1;
    wave::MatMulWaveOp::WaveId  m_WaveId;
    std::string                 m_WaveIdFormat          = "";
    // waveop name
    // waveop type
    kcc_int32                   m_WeightsAtomId         = -1;
    kcc_int32                   m_WeightsOffsetInAtom   = -1;

    // Pool
    kcc_int32                   m_DstSbAtomId       = -1;
    kcc_int32                   m_DstSbOffsetInAtom = -1;
    kcc_int32                   m_DstXNum           = -1;
    kcc_int32                   m_DstXStep	        = -1;
    kcc_int32                   m_DstYNum	        = -1;
    kcc_int32                   m_DstYStep	        = -1;
    kcc_int32                   m_DstZNum	        = -1;
    kcc_int32                   m_DstZStep	        = -1;
    std::string                 m_InDtype           = "";
    // "layername": "1conv/i1",
    kcc_int32                   m_NumPartitions     = -1;
    std::string                 m_OutDtype          = "";
    kcc_int32                   m_PoolFrequency	    = -1;
    std::string                 m_PoolFunc          = "";
    // previouswaveops": [ 1conv/i1/MatMuln0m0h0w0c0r0s0" ]
    bool                        m_SrcIsPsum         = true;
    kcc_int32                   m_SrcPsumBankId     = -1;
    kcc_int32                   m_SrcPsumBankOffset = -1;
    kcc_int32                   m_SrcSbAtomId       = -1;
    kcc_int32                   m_SrcSbOffsetInAtom = -1;
    kcc_int32                   m_SrcWNum	        = -1;
    kcc_int32                   m_SrcWStep	        = -1;
    kcc_int32                   m_SrcXNum	        = -1;
    kcc_int32                   m_SrcXStep	        = -1;
    kcc_int32                   m_SrcYNum	        = -1;
    kcc_int32                   m_SrcYStep	        = -1;
    kcc_int32                   m_SrcZNum	        = +1;
    kcc_int32                   m_SrcZStep	        = +1;
    std::vector<kcc_int32>      m_TileId;
    std::string                 m_TileIdFormat      = "";
    //waveopname": "1conv/i1/Pooln0m0h0w0",
    //waveoptype": "Pool"

    // Activation
    std::string                 m_ActivationFunc    = "";
    bool                        m_BiasAddEn         = false;
    kcc_int32                   m_BiasAtomId        = -1;
    kcc_int32                   m_BiasOffsetInAtom  = -1;
    kcc_int32                   m_DstPsumBankId     = -1;
#if 0
    kcc_int32                   m_DstXNum           = -1;
    kcc_int32                   m_DstXStep          = -1;
    kcc_int32                   m_DstYNum           = -1;
    kcc_int32                   m_DstYStep          = -1;
    kcc_int32                   m_DstZNum           = -1;
    kcc_int32                   m_DstZStep          = -1;
#endif
    //kcc_int32                   m_SrcPsumBankId     = -1;
#if 0
    kcc_int32                   m_SrcXNum           = -1;
    kcc_int32                   m_SrcXStep          = -1;
    kcc_int32                   m_SrcYNum           = -1;
    kcc_int32                   m_SrcYStep          = -1;
    kcc_int32                   m_SrcZNum           = +1;
    kcc_int32                   m_SrcZStep          = +1;
#endif
}; // class SerWaveOp



} // namespace serialize
} // namespace kcc

#endif // KCC_SERIALIZE_SERWAVEOP_H

