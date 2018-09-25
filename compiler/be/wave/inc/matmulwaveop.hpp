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

//#include "layers/inc/convlayer.hpp"

#include "wave/inc/waveop.hpp"


namespace kcc {

namespace wave {


class MatMulWaveOp : public WaveOp {
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

    kcc_int16 gPsumBankId() const {
        return m_PsumBankId;
    }

    kcc_int16 gPsumBankOffset () const {
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

    virtual WaveOpType gType() const override {
        return WaveOpType::MatMul;
    }

    bool verify() const override;


    kcc_int16 gFmapXNum () const {
        return m_FmapXNum;
    }

    kcc_int16 gFmapXStep () const {
        return m_FmapXStep;
    }

    kcc_int16 gFmapYNum () const {
        return m_FmapYNum;
    }

    kcc_int16 gFmapYStep () const {
        return m_FmapYStep;
    }

    kcc_int16 gFmapZNum () const {
        return m_FmapZNum;
    }

    kcc_int16 gFmapZStep () const {
        return m_FmapZStep;
    }

    kcc_int16 gPsumXNum () const {
        return m_PsumXNum;
    }

    kcc_int16 gPsumXStep () const {
        return m_PsumXStep;
    }

    kcc_int16 gPsumYNum () const {
        return m_PsumYNum;
    }

    kcc_int16 gPsumYStep () const {
        return m_PsumYStep;
    }

    kcc_int16 gPsumZNum () const {
        return m_PsumZNum;
    }

    kcc_int16 gPsumZStep () const {
        return m_PsumZStep;
    }

    kcc_int16 gNumColumnPartitions () const {
        return m_NumColumnPartitions;
    }

    kcc_int16 gNumRowPartitions () const {
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

private:
    MatMulWaveOp() = delete;
    MatMulWaveOp(const MatMulWaveOp&) = delete;

    MatMulWaveOp& operator= (const MatMulWaveOp&) const = delete;

private:
    kcc_int16       m_FmapXNum              = -1;
    kcc_int16       m_FmapXStep             = -1;
    kcc_int16       m_FmapYNum              = -1;
    kcc_int16       m_FmapYStep             = -1;
    kcc_int16       m_FmapZNum              = -1;
    kcc_int16       m_FmapZStep        = -1;
    kcc_int32       m_IfmapsSbAddress       = -1;
    const DataType& m_InDtype;
    // layer name
    kcc_int16       m_NumColumnPartitions   = -1;
    kcc_int16       m_NumRowPartitions      = -1;
    const DataType& m_OutDtype;
    // previous layers
    kcc_int16       m_PsumBankId            = -1;
    kcc_int16       m_PsumBankOffset        = -1;
    kcc_int16       m_PsumXNum              = -1;
    kcc_int16       m_PsumXStep             = -1;
    kcc_int16       m_PsumYNum              = -1;
    kcc_int16       m_PsumYStep             = -1;
    kcc_int16       m_PsumZNum              = -1;
    kcc_int16       m_PsumZStep             = -1;
    bool            m_StartTensorCalc       = true;
    bool            m_StopTensorCalc        = true;
    // waveop name
    // waveop type
    kcc_int32       m_WeightsSbAddress      = -2; // -1 means do not load weights

    kcc_int32       m_IfmapReplicationNumRows     = -1;
    kcc_int32       m_IfmapReplicationResolution  = -1;
    kcc_int32       m_IfmapReplicationShiftAmnt   = -1;
}; // class MatMulWaveOp : public WaveOp






class MatMulWaveOp::Params : public WaveOp::Params {
public:
    bool verify() const;
public:
    kcc_int16       m_FmapXNum              = -1;
    kcc_int16       m_FmapXStep             = -1;
    kcc_int16       m_FmapYNum              = -1;
    kcc_int16       m_FmapYStep             = -1;
    kcc_int16       m_FmapZNum              = -1;
    kcc_int16       m_FmapZStep             = -1;
    kcc_int32       m_IfmapsSbAddress       = -1;
    DataTypeId      m_InDtypeId             = DataTypeId::None;
    // layer name
    kcc_int16       m_NumColumnPartitions   = -1;
    kcc_int16       m_NumRowPartitions      = -1;
    DataTypeId      m_OutDtypeId            = DataTypeId::None;
    // previous layers
    kcc_int16       m_PsumBankId            = -1;
    kcc_int16       m_PsumBankOffset        = -1;
    kcc_int16       m_PsumXNum              = -1;
    kcc_int16       m_PsumXStep             = -1;
    kcc_int16       m_PsumYNum              = -1;
    kcc_int16       m_PsumYStep             = -1;
    kcc_int16       m_PsumZNum              = -1;
    kcc_int16       m_PsumZStep             = -1;
    bool            m_StartTensorCalc       = true;
    bool            m_StopTensorCalc        = true;
    // waveop name
    // waveop type
    kcc_int32       m_WeightsSbAddress      = -2;

    kcc_int32       m_IfmapReplicationNumRows     = -1;
    kcc_int32       m_IfmapReplicationResolution  = -1;
    kcc_int32       m_IfmapReplicationShiftAmnt   = -1;
};

}}


#endif


