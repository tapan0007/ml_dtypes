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

#include "events/inc/events.hpp"


namespace kcc {
using  namespace utils;

namespace serialize {


// common to all WaveOps
constexpr static const char* WaveOpKey_WaveOpType               = "waveop_type";
constexpr static const char* WaveOpKey_WaveOpName               = "waveop_name";
constexpr static const char* WaveOpKey_LayerName                = "layer_name";
constexpr static const char* WaveOpKey_PreviousWaveOps          = "previous_waveops";
constexpr static const char* WaveOpKey_PreviousEventIds         = "previous_event_ids";
constexpr static const char* WaveOpKey_PreviousEventWaitModes   = "previous_event_wait_modes";
constexpr static const char* WaveOpKey_PreviousEventSetModes    = "previous_event_set_modes";

constexpr static const char* WaveOpKey_Order                    = "order";


// MatMul
constexpr static const char* WaveOpKey_FmapXNum             = "src_x_num";
constexpr static const char* WaveOpKey_FmapXStep            = "src_x_step";
constexpr static const char* WaveOpKey_FmapYNum             = "src_y_num";
constexpr static const char* WaveOpKey_FmapYStep            = "src_y_step";
constexpr static const char* WaveOpKey_FmapZNum             = "src_z_num";
constexpr static const char* WaveOpKey_FmapZStep            = "src_z_step";
constexpr static const char* WaveOpKey_IfmapsSbAddress      = "src_sb_address";
constexpr static const char* WaveOpKey_DataType             = "data_type";

// MatMul
constexpr static const char* WaveOpKey_IfmapReplicationNumRows      = "ifmap_replication_num_rows";
constexpr static const char* WaveOpKey_IfmapReplicationResolution   = "ifmap_replication_resolution";
constexpr static const char* WaveOpKey_IfmapReplicationShiftAmnt    = "ifmap_replication_shift_amnt";

// SBAtomLoad
constexpr static const char* WaveOpKey_IfmapReplicationStepBytes    = "ifmap_replication_step_bytes";
constexpr static const char* WaveOpKey_SrcStepElem                  = "stride";

// layer name
constexpr static const char* WaveOpKey_NumColumnPartitions  = "num_column_partitions";
constexpr static const char* WaveOpKey_NumRowPartitions     = "num_row_partitions";
// previous waveops
constexpr static const char* WaveOpKey_PsumBankId           = "dst_psum_bank_id";
constexpr static const char* WaveOpKey_PsumBankOffset       = "dst_psum_bank_offset";
constexpr static const char* WaveOpKey_PsumXNum             = "dst_x_num";
constexpr static const char* WaveOpKey_PsumXStep            = "dst_x_step";
constexpr static const char* WaveOpKey_PsumYNum             = "dst_y_num";
constexpr static const char* WaveOpKey_PsumYStep            = "dst_y_step";
constexpr static const char* WaveOpKey_PsumZNum             = "dst_z_num";
constexpr static const char* WaveOpKey_PsumZStep            = "dst_z_step";
constexpr static const char* WaveOpKey_StartTensorCalc      = "start_tensor_calc";
constexpr static const char* WaveOpKey_StopTensorCalc       = "stop_tensor_calc";
// waveop name
// waveop type
constexpr static const char* WaveOpKey_WeightsSbAddress     = "weights_sb_address";

// SBAtom common
constexpr static const char* WaveOpKey_SbAddress            = "sb_address";
constexpr static const char* WaveOpKey_StartAtMidPart       = "start_at_mid_part";
constexpr static const char* WaveOpKey_Length               = "length";
constexpr static const char* WaveOpKey_OffsetInFile         = "offset_in_file";
constexpr static const char* WaveOpKey_PartitionStepBytes   = "partition_step_bytes";
constexpr static const char* WaveOpKey_RefFile              = "ref_file";
constexpr static const char* WaveOpKey_RefFileFormat        = "ref_file_format";
constexpr static const char* WaveOpKey_RefFileShape         = "ref_file_shape";

// SBAtomSave
constexpr static const char* WaveOpKey_FinalLayerOfmap      = "final_layer_ofmap";

// Pool
constexpr static const char* WaveOpKey_DstIsPsum            = "dst_is_psum";
constexpr static const char* WaveOpKey_DstSbAddress         = "dst_sb_address";
constexpr static const char* WaveOpKey_DstStartAtMidPart    = "dst_start_at_mid_part";
constexpr static const char* WaveOpKey_DstXNum              = "dst_x_num";
constexpr static const char* WaveOpKey_DstXStep             = "dst_x_step";
constexpr static const char* WaveOpKey_DstYNum              = "dst_y_num";
constexpr static const char* WaveOpKey_DstYStep             = "dst_y_step";
constexpr static const char* WaveOpKey_DstZNum              = "dst_z_num";
constexpr static const char* WaveOpKey_DstZStep             = "dst_z_step";
constexpr static const char* WaveOpKey_InDtype              = "in_dtype";
constexpr static const char* WaveOpKey_BiasDtype            = "bias_dtype";

constexpr static const char* WaveOpKey_NumPartitions        = "num_partitions";
constexpr static const char* WaveOpKey_OutDtype             = "out_dtype";
constexpr static const char* WaveOpKey_PoolFrequency        = "pool_frequency";
constexpr static const char* WaveOpKey_PoolFunc             = "pool_func";

constexpr static const char* WaveOpKey_SrcIsPsum            = "src_is_psum";
constexpr static const char* WaveOpKey_SrcPsumBankId        = "src_psum_bank_id";
constexpr static const char* WaveOpKey_SrcPsumBankOffset    = "src_psum_bank_offset";
constexpr static const char* WaveOpKey_SrcSbAddress         = "src_sb_address";
constexpr static const char* WaveOpKey_SrcStartAtMidPart    = "src_start_at_mid_part";
constexpr static const char* WaveOpKey_SrcWNum              = "src_w_num";
constexpr static const char* WaveOpKey_SrcWStep             = "src_w_step";
constexpr static const char* WaveOpKey_SrcXNum              = "src_x_num";
constexpr static const char* WaveOpKey_SrcXStep             = "src_x_step";
constexpr static const char* WaveOpKey_SrcYNum              = "src_y_num";
constexpr static const char* WaveOpKey_SrcYStep             = "src_y_step";
constexpr static const char* WaveOpKey_SrcZNum              = "src_z_num";
constexpr static const char* WaveOpKey_SrcZStep             = "src_z_step";

constexpr static const char* WaveOpKey_TileId               = "tile_id";
constexpr static const char* WaveOpKey_TileIdFormat         = "tile_id_format";

constexpr static const char* WaveOpKey_ActivationFunc              = "activation_func";
constexpr static const char* WaveOpKey_ActivationFunc_None         = "none"; /* until Jeff fixes none */
constexpr static const char* WaveOpKey_ActivationFunc_Identity     = "Identity";
constexpr static const char* WaveOpKey_ActivationFunc_Relu         = "Relu";
constexpr static const char* WaveOpKey_ActivationFunc_LeakyRelu    = "Lrelu";
constexpr static const char* WaveOpKey_ActivationFunc_Prelu        = "Prelu";
constexpr static const char* WaveOpKey_ActivationFunc_Sigmoid      = "Sigmoid";
constexpr static const char* WaveOpKey_ActivationFunc_Tanh         = "Tanh";
constexpr static const char* WaveOpKey_ActivationFunc_Exp          = "Exp";
constexpr static const char* WaveOpKey_ActivationFunc_Softplus     = "Softplus";


constexpr static const char* WaveOpKey_BiasAddEn            = "bias_add_en";
constexpr static const char* WaveOpKey_BiasSbAddress        = "bias_sb_address";
constexpr static const char* WaveOpKey_BiasStartAtMidPart   = "bias_start_at_mid_part";
constexpr static const char* WaveOpKey_DstPsumBankId        = "dst_psum_bank_id";
constexpr static const char* WaveOpKey_DstPsumBankOffset    = "dst_psum_bank_offset";


constexpr static const char* WaveOpKey_InADtype             = "in_a_dtype";
constexpr static const char* WaveOpKey_SrcAIsPsum           = "src_a_is_psum";
constexpr static const char* WaveOpKey_SrcAPsumBankId       = "src_a_psum_bank_id";
constexpr static const char* WaveOpKey_SrcAPsumBankOffset   = "src_a_psum_bank_offset";
constexpr static const char* WaveOpKey_SrcASbAddress        = "src_a_sb_address";
constexpr static const char* WaveOpKey_SrcAStartAtMidPart   = "src_a_start_at_mid_part";
constexpr static const char* WaveOpKey_SrcAWNum             = "src_a_w_num";
constexpr static const char* WaveOpKey_SrcAWStep            = "src_a_w_step";
constexpr static const char* WaveOpKey_SrcAXNum             = "src_a_x_num";
constexpr static const char* WaveOpKey_SrcAXStep            = "src_a_x_step";
constexpr static const char* WaveOpKey_SrcAYNum             = "src_a_y_num";
constexpr static const char* WaveOpKey_SrcAYStep            = "src_a_y_step";
constexpr static const char* WaveOpKey_SrcAZNum             = "src_a_z_num";
constexpr static const char* WaveOpKey_SrcAZStep            = "src_a_z_step";

constexpr static const char* WaveOpKey_InBDtype             = "in_b_dtype";
constexpr static const char* WaveOpKey_SrcBIsPsum           = "src_b_is_psum";
constexpr static const char* WaveOpKey_SrcBPsumBankId       = "src_b_psum_bank_id";
constexpr static const char* WaveOpKey_SrcBPsumBankOffset   = "src_b_psum_bank_offset";
constexpr static const char* WaveOpKey_SrcBSbAddress        = "src_b_sb_address";
constexpr static const char* WaveOpKey_SrcBStartAtMidPart   = "src_b_start_at_mid_part";
constexpr static const char* WaveOpKey_SrcBWNum             = "src_b_w_num";
constexpr static const char* WaveOpKey_SrcBWStep            = "src_b_w_step";
constexpr static const char* WaveOpKey_SrcBXNum             = "src_b_x_num";
constexpr static const char* WaveOpKey_SrcBXStep            = "src_b_x_step";
constexpr static const char* WaveOpKey_SrcBYNum             = "src_b_y_num";
constexpr static const char* WaveOpKey_SrcBYStep            = "src_b_y_step";
constexpr static const char* WaveOpKey_SrcBZNum             = "src_b_z_num";
constexpr static const char* WaveOpKey_SrcBZStep            = "src_b_z_step";

constexpr static const char* WaveOpKey_ContainWeights       = "contain_weights";
constexpr static const char* WaveOpKey_Engine               = "engine";


constexpr static const char* WaveOpKey_MinValue             = "min_val";
constexpr static const char* WaveOpKey_MaxValue             = "max_val";

constexpr static const char* WaveOpKey_Add             = "add";
constexpr static const char* WaveOpKey_Scale           = "scale";

constexpr static const char* WaveOpKey_MulScalar       = "mul_scalar";
constexpr static const char* WaveOpKey_AddScalar       = "add_scalar";



class SerWaveOp {
public:
    SerWaveOp();


    SerWaveOp(const SerWaveOp&) = default;

public:

    template<typename Archive>
    void save(Archive & archive) const;

    template<typename Archive>
    void load(Archive & archive);


    void addPreviousWaveOp(const std::string& prevWaveOp) { m_PreviousWaveOps.push_back(prevWaveOp);
    }
    void addPreviousEventId(events::EventId eventId) {
        m_PreviousEventIds.push_back(eventId);
    }
    void addPreviousEventWaitMode(events::EventWaitMode mode) {
        m_PreviousEventWaitModes.push_back(eventWaitMode2Isa(mode));
    }
    void addPrevEventSetMode(events::EventSetMode mode) {
        m_PreviousEventSetModes.push_back(eventSetMode2Isa(mode));
    }

    static ActivationFunc str2ActivationFunc(const std::string& s);
    static std::string activationType2Str(ActivationFunc);

private:
    void loadSrc(cereal::JSONInputArchive& archive, Dims dims);
    void loadSrcA(cereal::JSONInputArchive& archive, Dims dims);
    void loadSrcB(cereal::JSONInputArchive& archive, Dims dims);
    void loadSrcAB(cereal::JSONInputArchive& archive, Dims dims);
    void loadDst(cereal::JSONInputArchive& archive, Dims dims);

    void saveSrc(cereal::JSONOutputArchive& archive, Dims dims) const;
    void saveSrcA(cereal::JSONOutputArchive& archive, Dims dims) const;
    void saveSrcB(cereal::JSONOutputArchive& archive, Dims dims) const;
    void saveSrcAB(cereal::JSONOutputArchive& archive, Dims dims) const;
    void saveDst(cereal::JSONOutputArchive& archive, Dims dims) const;

    void loadSbAtom(cereal::JSONInputArchive& archive);
    void loadPool(cereal::JSONInputArchive& archive);
    void loadMatMul(cereal::JSONInputArchive& archive);
    void loadActivation(cereal::JSONInputArchive& archive);

    void loadResAdd(cereal::JSONInputArchive& archive);
    void loadScaleAdd(cereal::JSONInputArchive& archive);
    void loadAdd(cereal::JSONInputArchive& archive);
    void loadMultiply(cereal::JSONInputArchive& archive);
    void loadClipByValue(cereal::JSONInputArchive& archive);
    void loadMaximum(cereal::JSONInputArchive& archive);
    void loadMinimum(cereal::JSONInputArchive& archive);

    void saveSbAtom(cereal::JSONOutputArchive& archive) const;
    void savePool(cereal::JSONOutputArchive& archive) const;
    void saveMatMul(cereal::JSONOutputArchive& archive) const;
    void saveActivation(cereal::JSONOutputArchive& archive) const;
    void saveResAdd(cereal::JSONOutputArchive& archive) const;
    void saveScaleAdd(cereal::JSONOutputArchive& archive) const;
    void saveClipByValue(cereal::JSONOutputArchive& archive) const;
    void saveBarrier(cereal::JSONOutputArchive& archive) const;
    void saveNop(cereal::JSONOutputArchive& archive) const;
    void saveMaximum(cereal::JSONOutputArchive& archive) const;
    void saveMinimum(cereal::JSONOutputArchive& archive) const;

protected:
    bool verify() const;

private:
    bool verifySbAtom() const;
    bool verifySbAtomLoad() const;
    bool verifySbAtomSave() const;
    bool verifyMatMul() const;
    bool verifyPool() const;
    bool verifyActivation() const;
    bool verifyResAdd() const;
    bool verifyScaleAdd() const;
    bool verifyClipByValue() const;
    bool verifyBarrier() const;
    bool verifyNop() const;

public:
    // common to all
    std::string                 m_WaveOpType        = "";
    std::string                 m_WaveOpName        = "";
    std::string                 m_LayerName         = "";
    std::vector<std::string>    m_PreviousWaveOps;
    std::vector<int>            m_PreviousEventIds;
    std::vector<int>            m_PreviousEventWaitModes;
    std::vector<int>            m_PreviousEventSetModes;

    std::string                 m_Engine;

    // SBAtom
    kcc_int64                   m_SbAddress         = -1;
    bool                        m_StartAtMidPart    = false;
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

    // SBAtomLoad
    bool                        m_ContainWeights    = false;

    // SBAtomSave
    bool                        m_FinalLayerOfmap;

    // MatMul
    kcc_int32                   m_FmapXNum              = -1;
    kcc_int32                   m_FmapXStep             = -1;
    kcc_int32                   m_FmapYNum              = -1;
    kcc_int32                   m_FmapYStep             = -1;
    kcc_int32                   m_FmapZNum              = -1;
    kcc_int32                   m_FmapZStep             = -1;
    kcc_int64                   m_IfmapsSbAddress       = -1;
    // layer name
    kcc_int32                   m_NumColumnPartitions   = -1;
    kcc_int32                   m_NumRowPartitions      = -1;
    // previous waveops
    kcc_int32                   m_PsumBankId            = -1;
    kcc_int32                   m_PsumBankOffset        = -1;
    kcc_int32                   m_PsumXNum              = -1;
    kcc_int32                   m_PsumXStep             = -1;
    kcc_int32                   m_PsumYNum              = -1;
    kcc_int32                   m_PsumYStep             = -1;
    kcc_int32                   m_PsumZNum              = -1;
    kcc_int32                   m_PsumZStep             = -1;
    bool                        m_StartTensorCalc       = true;
    bool                        m_StopTensorCalc        = true;
    // waveop name
    // waveop type
    kcc_int64                   m_WeightsSbAddress      = -2;

    // Pool
    kcc_int64                   m_DstSbAddress      = -1;
    bool                        m_DstStartAtMidPart = false;
    kcc_int32                   m_DstXNum           = -1;
    kcc_int32                   m_DstXStep          = -1;
    kcc_int32                   m_DstYNum           = -1;
    kcc_int32                   m_DstYStep          = -1;
    kcc_int32                   m_DstZNum           = -1;
    kcc_int32                   m_DstZStep          = -1;
    std::string                 m_InDtype           = "";
    std::string                 m_BiasDtype         = "";
    // "layername": "1conv/i1",
    kcc_int32                   m_NumPartitions     = -1;
    std::string                 m_OutDtype          = "";
    kcc_int32                   m_PoolFrequency     = -1;
    std::string                 m_PoolFunc          = "";

    // previouswaveops": [ 1conv/i1/MatMuln0m0h0w0c0r0s0" ]
    bool                        m_SrcIsPsum         = true;
    kcc_int32                   m_SrcPsumBankId     = -1;
    kcc_int32                   m_SrcPsumBankOffset = -1;
    kcc_int64                   m_SrcSbAddress      = -1;
    bool                        m_SrcStartAtMidPart = false;
    kcc_int32                   m_SrcWNum           = -1;
    kcc_int32                   m_SrcWStep          = -1;
    kcc_int32                   m_SrcXNum           = -1;
    kcc_int32                   m_SrcXStep          = -1;
    kcc_int32                   m_SrcYNum           = -1;
    kcc_int32                   m_SrcYStep          = -1;
    kcc_int32                   m_SrcZNum           = +1;
    kcc_int32                   m_SrcZStep          = +1;

    std::string                 m_InADtype           = "";
    bool                        m_SrcAIsPsum            = true;
    kcc_int32                   m_SrcAPsumBankId        = -1;
    kcc_int32                   m_SrcAPsumBankOffset    = -1;
    kcc_int64                   m_SrcASbAddress         = -1;
    bool                        m_SrcAStartAtMidPart    = false;
    kcc_int32                   m_SrcAWNum              = -1;
    kcc_int32                   m_SrcAWStep             = -1;
    kcc_int32                   m_SrcAXNum              = -1;
    kcc_int32                   m_SrcAXStep             = -1;
    kcc_int32                   m_SrcAYNum              = -1;
    kcc_int32                   m_SrcAYStep             = -1;
    kcc_int32                   m_SrcAZNum              = -1;
    kcc_int32                   m_SrcAZStep             = -1;

    std::string                 m_InBDtype           = "";
    bool                        m_SrcBIsPsum            = true;
    kcc_int32                   m_SrcBPsumBankId        = -1;
    kcc_int32                   m_SrcBPsumBankOffset    = -1;
    kcc_int64                   m_SrcBSbAddress         = -1;
    bool                        m_SrcBStartAtMidPart    = false;
    kcc_int32                   m_SrcBWNum              = -1;
    kcc_int32                   m_SrcBWStep             = -1;
    kcc_int32                   m_SrcBXNum              = -1;
    kcc_int32                   m_SrcBXStep             = -1;
    kcc_int32                   m_SrcBYNum              = -1;
    kcc_int32                   m_SrcBYStep             = -1;
    kcc_int32                   m_SrcBZNum              = -1;
    kcc_int32                   m_SrcBZStep             = -1;

    std::vector<kcc_int32>      m_TileId;
    std::string                 m_TileIdFormat          = "";
    //waveopname": "1conv/i1/Pooln0m0h0w0",
    //waveoptype": "Pool"

    // Activation
    std::string                 m_ActivationFunc    = "";
    bool                        m_BiasAddEn         = false;
    kcc_int64                   m_BiasSbAddress     = -1;
    bool                        m_BiasStartAtMidPart = false;
    bool                        m_DstIsPsum         = true;
    kcc_int32                   m_DstPsumBankId     = -1;
    kcc_int32                   m_DstPsumBankOffset = -1;

    // MatMul and SbAtomLoad
    kcc_int32                   m_IfmapReplicationNumRows       = -1; // MM, Load
    kcc_int32                   m_IfmapReplicationResolution    = -1; // MM, Load
    kcc_int32                   m_IfmapReplicationShiftAmnt     = -1; // MatMul
    kcc_int32                   m_IfmapReplicationStepBytes     = -1; // SbAtomLoad

    kcc_int32                   m_SrcStepElem                   = -1; // SbAtomLoad

    kcc_float32                 m_MinValue;
    kcc_float32                 m_MaxValue;

    kcc_float32                 m_AddScalar = 0.0;
    kcc_float32                 m_MulScalar = 0.0;
    kcc_float32                 m_Add;
    kcc_float32                 m_Scale;

    kcc_int32                   m_Order                         = -1;
}; // class SerWaveOp



} // namespace serialize
} // namespace kcc

#endif // KCC_SERIALIZE_SERWAVEOP_H

