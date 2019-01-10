#pragma once

#ifndef KCC_WAVE_MATMULWAVEOP_H
#define KCC_WAVE_MATMULWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>





#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"


#include "wave/inc/waveop.hpp"


namespace kcc {

namespace wave {


class MatMulWaveOp : public WaveOp {
private:
    using BaseClass = WaveOp;
public:
    class Params;

public:
    MatMulWaveOp(const MatMulWaveOp::Params& params,
        const std::vector<WaveOp*>& prevWaveOps);

    kcc_int32 gIfmapsSbAddress() const {
        return m_IfmapsSbAddress;
    }

    kcc_int32 gWeightsSbAddress() const {
        return m_WeightsSbAddress;
    }

    kcc_int32 gPsumBankId() const {
        return m_PsumBankId;
    }

    kcc_int32 gPsumBankOffset () const {
        return m_PsumBankOffset;
    }

    bool qStartTensorCalc() const {
        return m_StartTensorCalc;
    }

    bool qStopTensorCalc() const {
        return m_StopTensorCalc;
    }

    //----------------------------------------------------------------
    bool qMatMulWaveOp() const override {
        return true;
    }

    EngineId gEngineId() const override {
        return EngineId::PeArray;
    }

    static std::string gTypeStrStatic();

    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    WaveOpType gType() const override {
        return WaveOpType::MatMul;
    }

    bool verify() const override;


    kcc_int32 gFmapXNum () const {
        return m_FmapXNum;
    }

    kcc_int32 gFmapXStep () const {
        return m_FmapXStep;
    }

    kcc_int32 gFmapYNum () const {
        return m_FmapYNum;
    }

    kcc_int32 gFmapYStep () const {
        return m_FmapYStep;
    }

    kcc_int32 gFmapZNum () const {
        return m_FmapZNum;
    }

    kcc_int32 gFmapZStep () const {
        return m_FmapZStep;
    }

    kcc_int32 gPsumXNum () const {
        return m_PsumXNum;
    }

    kcc_int32 gPsumXStep () const {
        return m_PsumXStep;
    }

    kcc_int32 gPsumYNum () const {
        return m_PsumYNum;
    }

    kcc_int32 gPsumYStep () const {
        return m_PsumYStep;
    }

    kcc_int32 gPsumZNum () const {
        return m_PsumZNum;
    }

    kcc_int32 gPsumZStep () const {
        return m_PsumZStep;
    }

    kcc_int32 gNumColumnPartitions () const {
        return m_NumColumnPartitions;
    }

    kcc_int32 gNumRowPartitions () const {
        return m_NumRowPartitions;
    }

    const DataType& gInDtype () const {
        return m_InDtype;
    }
    const DataType& gOutDtype () const {
        return m_OutDtype;
    }

    kcc_int32 gIfmapReplicationNumRows() const {
        return m_IfmapReplicationNumRows;
    }
    kcc_int32 gIfmapReplicationResolution() const {
        return m_IfmapReplicationResolution;
    }
    kcc_int32 gIfmapReplicationShiftAmnt() const {
        return m_IfmapReplicationShiftAmnt;
    }
    kcc_uint16 gQuantOffsetIfmaps() const {
        return m_QuantOffsetIfmaps;
    }
    kcc_uint16 gQuantOffsetWeights() const {
        return m_QuantOffsetWeights;
    }
    PEPerfOptType gPEPerfOptMode() const {
        return m_PEPerfOptMode;
    }

    bool qIsDynamicWeights() const {
        return m_IsDynamicWeights;
    }

private:
    MatMulWaveOp() = delete;
    MatMulWaveOp(const MatMulWaveOp&) = delete;

    MatMulWaveOp& operator= (const MatMulWaveOp&) const = delete;

private:
    kcc_int32       m_FmapXNum              = -1;
    kcc_int32       m_FmapXStep             = -1;
    kcc_int32       m_FmapYNum              = -1;
    kcc_int32       m_FmapYStep             = -1;
    kcc_int32       m_FmapZNum              = -1;
    kcc_int32       m_FmapZStep        = -1;
    kcc_int32       m_IfmapsSbAddress       = -1;
    const DataType& m_InDtype;
    kcc_int32       m_NumColumnPartitions   = -1;
    kcc_int32       m_NumRowPartitions      = -1;
    const DataType& m_OutDtype;
    kcc_int32       m_PsumBankId            = -1;
    kcc_int32       m_PsumBankOffset        = -1;
    kcc_int32       m_PsumXNum              = -1;
    kcc_int32       m_PsumXStep             = -1;
    kcc_int32       m_PsumYNum              = -1;
    kcc_int32       m_PsumYStep             = -1;
    kcc_int32       m_PsumZNum              = -1;
    kcc_int32       m_PsumZStep             = -1;
    bool            m_StartTensorCalc       = true;
    bool            m_StopTensorCalc        = true;
    // waveop name
    // waveop type
    kcc_int32       m_WeightsSbAddress      = -2; // -1 means do not load weights

    kcc_int32       m_IfmapReplicationNumRows     = -1;
    kcc_int32       m_IfmapReplicationResolution  = -1;
    kcc_int32       m_IfmapReplicationShiftAmnt   = -1;

    kcc_uint16      m_QuantOffsetIfmaps     = 0;
    kcc_uint16      m_QuantOffsetWeights    = 0;
    PEPerfOptType   m_PEPerfOptMode         = PEPerfOptType::None;

    bool            m_IsDynamicWeights      = false;
}; // class MatMulWaveOp : public WaveOp






class MatMulWaveOp::Params : public WaveOp::Params {
public:
    bool verify() const;
public:
    kcc_int32       m_FmapXNum              = -1;
    kcc_int32       m_FmapXStep             = -1;
    kcc_int32       m_FmapYNum              = -1;
    kcc_int32       m_FmapYStep             = -1;
    kcc_int32       m_FmapZNum              = -1;
    kcc_int32       m_FmapZStep             = -1;
    kcc_int32       m_IfmapsSbAddress       = -1;
    DataTypeId      m_InDtypeId             = DataTypeId::None;
    kcc_int32       m_NumColumnPartitions   = -1;
    kcc_int32       m_NumRowPartitions      = -1;
    DataTypeId      m_OutDtypeId            = DataTypeId::None;
    kcc_int32       m_PsumBankId            = -1;
    kcc_int32       m_PsumBankOffset        = -1;
    kcc_int32       m_PsumXNum              = -1;
    kcc_int32       m_PsumXStep             = -1;
    kcc_int32       m_PsumYNum              = -1;
    kcc_int32       m_PsumYStep             = -1;
    kcc_int32       m_PsumZNum              = -1;
    kcc_int32       m_PsumZStep             = -1;
    bool            m_StartTensorCalc       = true;
    bool            m_StopTensorCalc        = true;
    // waveop name
    // waveop type
    kcc_int32       m_WeightsSbAddress      = -2;

    kcc_int32       m_IfmapReplicationNumRows     = -1;
    kcc_int32       m_IfmapReplicationResolution  = -1;
    kcc_int32       m_IfmapReplicationShiftAmnt   = -1;

    kcc_uint16      m_QuantOffsetIfmaps     = 0;
    kcc_uint16      m_QuantOffsetWeights    = 0;
    PEPerfOptType   m_PEPerfOptMode         = PEPerfOptType::None;

    bool            m_IsDynamicWeights      = false;
};

}}


#endif


