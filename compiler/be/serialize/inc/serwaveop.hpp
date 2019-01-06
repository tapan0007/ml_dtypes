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
#include "utils/inc/asserter.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/types.hpp"
#include "utils/inc/datatype.hpp"

#include "events/inc/events.hpp"


namespace kcc {
using  namespace utils;

namespace serialize {


// common to all WaveOps
namespace WaveOpKey {
constexpr static const char* WaveOpType               = "waveop_type";
constexpr static const char* WaveOpName               = "waveop_name";
constexpr static const char* LayerName                = "layer_name";
constexpr static const char* PreviousWaveOps          = "previous_waveops";

constexpr static const char* PreviousSyncs            = "previous_syncs";
constexpr static const char* SemaphoreSync            = "semaphore_sync";
constexpr static const char* EventSync                = "event_sync";

constexpr static const char* Order                    = "order";


// MatMul
constexpr static const char* FmapXNum             = "src_x_num";
constexpr static const char* FmapXStep            = "src_x_step";
constexpr static const char* FmapYNum             = "src_y_num";
constexpr static const char* FmapYStep            = "src_y_step";
constexpr static const char* FmapZNum             = "src_z_num";
constexpr static const char* FmapZStep            = "src_z_step";
constexpr static const char* IfmapsSbAddress      = "src_sb_address";
constexpr static const char* DataType             = "data_type";

// MatMul
constexpr static const char* IfmapReplicationNumRows      = "ifmap_replication_num_rows";
constexpr static const char* IfmapReplicationResolution   = "ifmap_replication_resolution";
constexpr static const char* IfmapReplicationShiftAmnt    = "ifmap_replication_shift_amnt";

constexpr static const char* ParallelMode                 = "parallel_mode";

// SBAtomLoad
constexpr static const char* IfmapReplicationStepBytes    = "ifmap_replication_step_bytes";
constexpr static const char* SrcStepElem                  = "stride";

constexpr static const char* NumColumnPartitions  = "num_column_partitions";
constexpr static const char* NumRowPartitions     = "num_row_partitions";
// previous waveops
constexpr static const char* PsumBankId           = "dst_psum_bank_id";
constexpr static const char* PsumBankOffset       = "dst_psum_bank_offset";
constexpr static const char* PsumXNum             = "dst_x_num";
constexpr static const char* PsumXStep            = "dst_x_step";
constexpr static const char* PsumYNum             = "dst_y_num";
constexpr static const char* PsumYStep            = "dst_y_step";
constexpr static const char* PsumZNum             = "dst_z_num";
constexpr static const char* PsumZStep            = "dst_z_step";
constexpr static const char* StartTensorCalc      = "start_tensor_calc";
constexpr static const char* StopTensorCalc       = "stop_tensor_calc";
// waveop name
// waveop type
constexpr static const char* WeightsSbAddress     = "weights_sb_address";

// SBAtom common
constexpr static const char* SbAddress            = "sb_address";
constexpr static const char* StartAtMidPart       = "start_at_mid_part";
constexpr static const char* Length               = "length";
constexpr static const char* OffsetInFile         = "offset_in_file";
constexpr static const char* PartitionStepBytes   = "partition_step_bytes";
constexpr static const char* RefFile              = "ref_file";
constexpr static const char* RefFileFormat        = "ref_file_format";
constexpr static const char* RefFileShape         = "ref_file_shape";

// SBAtomSave
constexpr static const char* FinalLayerOfmap      = "final_layer_ofmap";

// Pool
constexpr static const char* DstIsPsum            = "dst_is_psum";
constexpr static const char* DstSbAddress         = "dst_sb_address";
constexpr static const char* DstStartAtMidPart    = "dst_start_at_mid_part";
constexpr static const char* DstXNum              = "dst_x_num";
constexpr static const char* DstXStep             = "dst_x_step";
constexpr static const char* DstYNum              = "dst_y_num";
constexpr static const char* DstYStep             = "dst_y_step";
constexpr static const char* DstZNum              = "dst_z_num";
constexpr static const char* DstZStep             = "dst_z_step";
constexpr static const char* InDtype              = "in_dtype";
constexpr static const char* BiasDtype            = "bias_dtype";

constexpr static const char* NumPartitions        = "num_partitions";
constexpr static const char* OutDtype             = "out_dtype";
constexpr static const char* PoolFrequency        = "pool_frequency";
constexpr static const char* PoolFunc             = "pool_func";

constexpr static const char* SrcIsPsum            = "src_is_psum";
constexpr static const char* SrcPsumBankId        = "src_psum_bank_id";
constexpr static const char* SrcPsumBankOffset    = "src_psum_bank_offset";
constexpr static const char* SrcSbAddress         = "src_sb_address";
constexpr static const char* SrcStartAtMidPart    = "src_start_at_mid_part";
constexpr static const char* SrcWNum              = "src_w_num";
constexpr static const char* SrcWStep             = "src_w_step";
constexpr static const char* SrcXNum              = "src_x_num";
constexpr static const char* SrcXStep             = "src_x_step";
constexpr static const char* SrcYNum              = "src_y_num";
constexpr static const char* SrcYStep             = "src_y_step";
constexpr static const char* SrcZNum              = "src_z_num";
constexpr static const char* SrcZStep             = "src_z_step";

constexpr static const char* TileId               = "tile_id";
constexpr static const char* TileIdFormat         = "tile_id_format";

constexpr static const char* ActivationFunc              = "activation_func";
constexpr static const char* ActivationFunc_None         = "none"; /* until Jeff fixes none */
constexpr static const char* ActivationFunc_Identity     = "Identity";
constexpr static const char* ActivationFunc_Relu         = "Relu";
constexpr static const char* ActivationFunc_LeakyRelu    = "Lrelu";
constexpr static const char* ActivationFunc_Prelu        = "Prelu";
constexpr static const char* ActivationFunc_Sigmoid      = "Sigmoid";
constexpr static const char* ActivationFunc_Tanh         = "Tanh";
constexpr static const char* ActivationFunc_Exp          = "Exp";
constexpr static const char* ActivationFunc_Softplus     = "Softplus";
constexpr static const char* ActivationFunc_Sqrt         = "Sqrt";


constexpr static const char* BiasAddEn            = "bias_add_en";
constexpr static const char* BiasSbAddress        = "bias_sb_address";
constexpr static const char* BiasStartAtMidPart   = "bias_start_at_mid_part";
constexpr static const char* DstPsumBankId        = "dst_psum_bank_id";
constexpr static const char* DstPsumBankOffset    = "dst_psum_bank_offset";


constexpr static const char* InADtype             = "in_a_dtype";
constexpr static const char* SrcAIsPsum           = "src_a_is_psum";
constexpr static const char* SrcAPsumBankId       = "src_a_psum_bank_id";
constexpr static const char* SrcAPsumBankOffset   = "src_a_psum_bank_offset";
constexpr static const char* SrcASbAddress        = "src_a_sb_address";
constexpr static const char* SrcAStartAtMidPart   = "src_a_start_at_mid_part";
constexpr static const char* SrcAWNum             = "src_a_w_num";
constexpr static const char* SrcAWStep            = "src_a_w_step";
constexpr static const char* SrcAXNum             = "src_a_x_num";
constexpr static const char* SrcAXStep            = "src_a_x_step";
constexpr static const char* SrcAYNum             = "src_a_y_num";
constexpr static const char* SrcAYStep            = "src_a_y_step";
constexpr static const char* SrcAZNum             = "src_a_z_num";
constexpr static const char* SrcAZStep            = "src_a_z_step";

constexpr static const char* InBDtype             = "in_b_dtype";
constexpr static const char* SrcBIsPsum           = "src_b_is_psum";
constexpr static const char* SrcBPsumBankId       = "src_b_psum_bank_id";
constexpr static const char* SrcBPsumBankOffset   = "src_b_psum_bank_offset";
constexpr static const char* SrcBSbAddress        = "src_b_sb_address";
constexpr static const char* SrcBStartAtMidPart   = "src_b_start_at_mid_part";
constexpr static const char* SrcBWNum             = "src_b_w_num";
constexpr static const char* SrcBWStep            = "src_b_w_step";
constexpr static const char* SrcBXNum             = "src_b_x_num";
constexpr static const char* SrcBXStep            = "src_b_x_step";
constexpr static const char* SrcBYNum             = "src_b_y_num";
constexpr static const char* SrcBYStep            = "src_b_y_step";
constexpr static const char* SrcBZNum             = "src_b_z_num";
constexpr static const char* SrcBZStep            = "src_b_z_step";

constexpr static const char* ContainWeights       = "contain_weights";
constexpr static const char* Engine               = "engine";


constexpr static const char* MinValue             = "min_val";
constexpr static const char* MaxValue             = "max_val";

constexpr static const char* Add                  = "add";
constexpr static const char* Scale                = "scale";

constexpr static const char* QuantOffsetIfmaps    = "quant_offset_ifmaps";
constexpr static const char* QuantOffsetWeights   = "quant_offset_weights";
constexpr static const char* PEPerfOptMode        = "pe_perf_opt_mode";

//constexpr static const char* MulScalar            = "mul_scalar";
//constexpr static const char* AddScalar            = "add_scalar";
constexpr static const char* IsScalarOp           = "is_scalar_op";
constexpr static const char* ScalarVal            = "scalar_val";

constexpr static const char* Op                   = "op";
constexpr static const char* Op0                  = "op0";
constexpr static const char* Op1                  = "op1";
constexpr static const char* ImmVal0              = "imm_val0";
constexpr static const char* ImmVal1              = "imm_val1";
constexpr static const char* ImmPtr0              = "imm_ptr0";
constexpr static const char* ImmPtr1              = "imm_ptr1";
constexpr static const char* IsDynamicWeights     = "is_dynamic_weights";


constexpr static const char* PairLoadWaveOp    = "pair_load";
constexpr static const char* PrevCopyWaveOp    = "prev_copy";
constexpr static const char* PairCopyWaveOp    = "pair_copy";
constexpr static const char* SizeInBytes       = "size_in_bytes";
//constexpr static const char* WaveOpKey_EngineId          = EngineId::None;
} // namespace WaveOpKey

//===================================================
class SerWaveOp {
public:
    SerWaveOp();


    SerWaveOp(const SerWaveOp&) = default;

public:

    template<typename Archive>
    void save(Archive & archive) const;

    template<typename Archive>
    void load(Archive & archive);

private:
    class Sync;

public:
    void addPreviousWaveOp(const std::string& prevWaveOp) {
        m_PreviousWaveOps.push_back(prevWaveOp);
    }

    void addPreviousEventSync(events::EventSetMode setMode,
                              events::EventId eventId,
                              events::EventWaitMode waitMode);
    void addPreviousSemaphoreSync(const char* prevSemaphore, kcc_int32 trigOrd);
    void addPreviousSemaphoreSync(const char* prevSemaphore, kcc_int32 trigOrd,
                                  const char* prevSemaphore1, kcc_int32 trigOrd1);


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
    void loadReciprocal(cereal::JSONInputArchive& archive);
    void loadRegLoad(cereal::JSONInputArchive& archive);
    void loadRegStore(cereal::JSONInputArchive& archive);
    void loadMatMul(cereal::JSONInputArchive& archive);
    void loadActivation(cereal::JSONInputArchive& archive);

    void loadResAdd(cereal::JSONInputArchive& archive);
    void loadScaleAdd(cereal::JSONInputArchive& archive);
    void loadAdd(cereal::JSONInputArchive& archive);
    void loadSub(cereal::JSONInputArchive& archive);
    void loadMultiply(cereal::JSONInputArchive& archive);
    void loadClipByValue(cereal::JSONInputArchive& archive);
    void loadMaximum(cereal::JSONInputArchive& archive);
    void loadMinimum(cereal::JSONInputArchive& archive);
    void loadTensorTensor(cereal::JSONInputArchive& archive);
    void loadTensorScalar(cereal::JSONInputArchive& archive);
    void loadTensorScalarPtr(cereal::JSONInputArchive& archive);


    void saveSbAtom(cereal::JSONOutputArchive& archive) const;
    void savePool(cereal::JSONOutputArchive& archive) const;
    void saveTpbCopy(cereal::JSONOutputArchive& archive) const;
    void saveReciprocal(cereal::JSONOutputArchive& archive) const;
    void saveRegLoad(cereal::JSONOutputArchive& archive) const;
    void saveRegStore(cereal::JSONOutputArchive& archive) const;
    void saveMatMul(cereal::JSONOutputArchive& archive) const;
    void saveActivation(cereal::JSONOutputArchive& archive) const;
    void saveResAdd(cereal::JSONOutputArchive& archive) const;
    void saveScaleAdd(cereal::JSONOutputArchive& archive) const;
    void saveClipByValue(cereal::JSONOutputArchive& archive) const;
    void saveNop(cereal::JSONOutputArchive& archive) const;
    void saveMaximum(cereal::JSONOutputArchive& archive) const;
    void saveMinimum(cereal::JSONOutputArchive& archive) const;
    void saveAdd(cereal::JSONOutputArchive& archive) const;
    void saveSub(cereal::JSONOutputArchive& archive) const;
    void saveMult(cereal::JSONOutputArchive& archive) const;
    void saveTensorTensor(cereal::JSONOutputArchive& archive) const;
    void saveTensorScalar(cereal::JSONOutputArchive& archive) const;
    void saveTensorScalarPtr(cereal::JSONOutputArchive& archive) const;

protected:
    bool verify() const;

private:
    bool verifySbAtom() const;
    bool verifySbAtomLoad() const;
    bool verifySbAtomSave() const;
    bool verifyMatMul() const;
    bool verifyPool() const;
    bool verifyReciprocal() const;
    bool verifyRegLoad() const;
    bool verifyRegStore() const;
    bool verifyActivation() const;
    bool verifyResAdd() const;
    bool verifyScaleAdd() const;
    bool verifyClipByValue() const;
    bool verifyNop() const;
    bool verifyTensor() const;
    bool verifyTensorTensor() const;
    bool verifyTensorScalar() const;
    bool verifyTpbCopy() const;

public:
    // common to all
    std::string                 m_WaveOpType        = "";
    std::string                 m_WaveOpName        = "";
    std::string                 m_LayerName         = "";
    std::vector<std::string>    m_PreviousWaveOps;
    std::vector<Sync>           m_PreviousSyncs;

    std::string                 m_Engine;

    // SBAtom
    kcc_int64                   m_SbAddress         = -1;
    bool                        m_StartAtMidPart    = false;
    std::string                 m_DataType          = "";
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
    std::string                 m_PairCopyWaveOp    = "";

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
    bool                        m_IsDynamicWeights     = false;

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
    kcc_int32                   m_NumPartitions     = -1;
    std::string                 m_OutDtype          = "";
    kcc_int32                   m_PoolFrequency     = -1;
    std::string                 m_PoolFunc          = "";

    std::string                 m_PairLoadWaveOp    = "";
    std::string                 m_PrevCopyWaveOp    = "";
    kcc_int64                   m_SrcAddress        = -1;
    kcc_int64                   m_DstAddress        = -1;
    kcc_int64                   m_SizeInBytes       = -1;

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

    bool                        m_ParallelMode;

    std::string                 m_Op;
    std::string                 m_Op0;
    std::string                 m_Op1;
    kcc_float32                 m_ImmVal0;
    kcc_float32                 m_ImmVal1;
    TpbAddress                  m_ImmPtr0;
    TpbAddress                  m_ImmPtr1;

    // ClipByValue
    kcc_float32                 m_MinValue;
    kcc_float32                 m_MaxValue;

    //kcc_float32                 m_AddScalar = 0.0;
    //kcc_float32                 m_MulScalar = 0.0;

    // Min,Max,Add,Sub,Mult
    bool                        m_IsScalarOp;
    kcc_float32                 m_ScalarVal;

    // ScaleAdd
    kcc_float32                 m_Add;
    kcc_float32                 m_Scale;

    // Quantize
    kcc_uint16                  m_QuantOffsetIfmaps             = 0;
    kcc_uint16                  m_QuantOffsetWeights            = 0;
    std::string                 m_PEPerfOptMode                 = PEPerfOptTypeStr::None;

    // Everyone
    kcc_int32                   m_Order                         = -1;
}; // class SerWaveOp





//===================================================
class SerWaveOp::Sync {
private:
    class EventSync {
    public:
        EventSync(events::EventSetMode setMode, events::EventId eventId, events::EventWaitMode waitMode)
            : m_SetMode(setMode)
            , m_EventId(eventId)
            , m_WaitMode(waitMode)
        {}
        EventSync(const EventSync&) = default;
        EventSync() = delete;
        ~EventSync() = default;
    public:
        const events::EventSetMode  m_SetMode  = events::EventSetMode::DontSet;
        const events::EventId       m_EventId  = 0;
        const events::EventWaitMode m_WaitMode = events::EventWaitMode::DontWait;
    };

    class SemSync {
    public:
        SemSync(const char* que, kcc_int32 trigOrd)
            : m_QueueName(que ? que : "")
            , m_TrigOrd(trigOrd)
            , m_QueueName1("")
            , m_TrigOrd1(-1)
        {}
        SemSync(const char* que, kcc_int32 trigOrd, const char* que1, kcc_int32 trigOrd1)
            : m_QueueName(que ? que : "")
            , m_TrigOrd(trigOrd)
            , m_QueueName1(que1 ? que1 : "")
            , m_TrigOrd1(trigOrd1)
        {}
        SemSync(const SemSync&) = default;
        SemSync() = delete;
        ~SemSync() = default;

    public:
        const std::string   m_QueueName  = "";
        const kcc_int32     m_TrigOrd    = -1;
        const std::string   m_QueueName1 = "";
        const kcc_int32     m_TrigOrd1   = -1;
    };

public:
    Sync(events::EventSetMode setMode, events::EventId eventId, events::EventWaitMode waitMode);
    Sync(const char* que, kcc_int32 trigOrd);
    Sync(const char* que, kcc_int32 trigOrd, const char* que1, kcc_int32 trigOrd1);
    Sync(const Sync& rhs);
    ~Sync();

    void save(cereal::JSONOutputArchive& archive) const;

private:
    bool m_WithEvent;
    union {
        EventSync m_EventSync;
        SemSync   m_SemSync;
    };
}; // SerWaveOp::Sync



//===================================================


} // namespace serialize
} // namespace kcc

#endif // KCC_SERIALIZE_SERWAVEOP_H

