#pragma once

#ifndef KCC_WAVE_ACTIVATIONWAVEOP_H
#define KCC_WAVE_ACTIVATIONWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>
#include <array>


#include "aws_tonga_isa_tpb_common.h"



#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"

#include "layers/inc/activlayer.hpp"

#include "wave/inc/waveop.hpp"


namespace kcc {

namespace wave {


class ActivationWaveOp : public WaveOp {
public:
    class Params;
public:
    ActivationWaveOp(const ActivationWaveOp::Params& params,
                           const std::vector<WaveOp*>& prevWaveOps);
public:
    bool verify() const override;

private:
    ActivationWaveOp() = delete;

public:
    // In the first version (Resnet50 support) for Activation SW supports only
    // the following combination of IFMAP input, BIAS input, OFMAP output srcs and dsts:
    //      IFMAP-src   BIAS-src   OFMAP-dst
    //      PSUM        SB         PSUM

    ActivationFunc gActivationFunc () const {
        return m_ActivationFunc;
    }
    TONGA_ISA_TPB_ACTIVATION_FUNC gSimActivationFunc() const;

    bool qBiasAddEn () const {
        return m_BiasAddEn;
    }
    kcc_int64 gBiasSbAddress () const {
        return m_BiasSbAddress;
    }
    bool gBiasStartAtMidPart () const {
        return m_BiasStartAtMidPart;
    }
    bool qActivationWaveOp() const override {
        return true;
    }
    EngineId gEngineId() const override {
        return EngineId::Activation;
    }
    static std::string gTypeStrStatic();

    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    virtual WaveOpType gType() const override {
        return WaveOpType::Activation;
    }

    kcc_int32 gDstPsumBankId() const {
        return m_DstPsumBankId;
    }

    kcc_int32 gDstPsumBankOffset() const {
        return m_DstPsumBankOffset;
    }

    kcc_int32 gDstXNum () const {
        return m_DstXNum;
    }
    kcc_int32 gDstXStep () const {
        return m_DstXStep;
    }
    kcc_int32 gDstYNum () const {
        return m_DstYNum;
    }
    kcc_int32 gDstYStep () const {
        return m_DstYStep;
    }
    kcc_int32 gDstZNum () const {
        return m_DstZNum;
    }
    kcc_int32 gDstZStep () const {
        return m_DstZStep;
    }
    const DataType& gInDtype () const {
        return m_InDtype;
    }
    const DataType& gBiasDtype () const {
        return m_BiasDtype;
    }
    kcc_int32 gNumPartitions () const {
        return m_NumPartitions;
    }
    const DataType& gOutDtype () const {
        return m_OutDtype;
    }
    bool qSrcIsPsum () const {
        return m_SrcIsPsum;
    }
    kcc_int64 gSrcSbAddress () const {
        return m_SrcSbAddress;
    }
    bool gSrcStartAtMidPart () const {
        return m_SrcStartAtMidPart;
    }
    kcc_int32 gSrcPsumBankId () const {
        return m_SrcPsumBankId;
    }
    kcc_int32 gSrcPsumBankOffset () const {
        return m_SrcPsumBankOffset;
    }
    kcc_int32 gSrcXNum () const {
        return m_SrcXNum;
    }
    kcc_int32 gSrcXStep () const {
        return m_SrcXStep;
    }
    kcc_int32 gSrcYNum () const {
        return m_SrcYNum;
    }
    kcc_int32 gSrcYStep () const {
        return m_SrcYStep;
    }
    kcc_int32 gSrcZNum () const {
        return m_SrcZNum;
    }
    kcc_int32 gSrcZStep () const {
        return m_SrcZStep;
    }

    const std::string& gTileIdFormat () const {
        return m_TileIdFormat;
    }
    const std::array<kcc_int32, 4>& gTileId () const {
        return m_TileId;
    }

    kcc_float32 gScale() const {
        return m_Scale;
    }

    bool qDstIsPsum () const {
        return m_DstIsPsum;
    }
    kcc_int64 gDstSbAddress () const {
        return m_DstSbAddress;
    }
    bool gDstStartAtMidPart () const {
        return m_DstStartAtMidPart;
    }

private:
    ActivationFunc              m_ActivationFunc        = ActivationFunc::Invalid;
    bool                        m_BiasAddEn;
    kcc_int64                   m_BiasSbAddress         = -1;
    bool                        m_BiasStartAtMidPart    = false;
    kcc_float32                 m_Scale                 = -1.0;
    bool                        m_DstIsPsum;
    kcc_int32                   m_DstPsumBankId         = -1;
    kcc_int32                   m_DstPsumBankOffset     = -1;
    kcc_int32                   m_DstXNum               = -1;
    kcc_int32                   m_DstXStep              = -1;
    kcc_int32                   m_DstYNum               = -1;
    kcc_int32                   m_DstYStep              = -1;
    kcc_int32                   m_DstZNum               = -1;
    kcc_int32                   m_DstZStep              = -1;
    kcc_int64                   m_DstSbAddress          = -1;
    bool                        m_DstStartAtMidPart     = false;
    const DataType&             m_InDtype;
    const DataType&             m_BiasDtype;
    kcc_int32                   m_NumPartitions         = -1;
    const DataType&             m_OutDtype;
    bool                        m_SrcIsPsum;
    kcc_int32                   m_SrcPsumBankId         = -1;
    kcc_int32                   m_SrcPsumBankOffset     = -1;
    kcc_int32                   m_SrcXNum               = -1;
    kcc_int32                   m_SrcXStep              = -1;
    kcc_int32                   m_SrcYNum               = -1;
    kcc_int32                   m_SrcYStep              = -1;
    kcc_int32                   m_SrcZNum               = 1; // until resolution of SIM
    kcc_int32                   m_SrcZStep              = 1; // //issues.amazon.com/issues/kaena-198
    kcc_int64                   m_SrcSbAddress          = -1;
    bool                        m_SrcStartAtMidPart     = false;
    std::array<kcc_int32, 4>    m_TileId;
    std::string                 m_TileIdFormat          = "";
}; // class ActivationWaveOp : public WaveOp




class ActivationWaveOp::Params : public WaveOp::Params {
public:
    bool verify() const;
public:
    ActivationFunc              m_ActivationFunc        = ActivationFunc::Invalid;
    bool                        m_BiasAddEn;
    kcc_int64                   m_BiasSbAddress         = -1;
    bool                        m_BiasStartAtMidPart    = false;
    kcc_float32                 m_Scale                 = -1.0;
    bool                        m_DstIsPsum;
    kcc_int32                   m_DstPsumBankId         = -1;
    kcc_int32                   m_DstPsumBankOffset     = -1;
    kcc_int32                   m_DstXNum               = -1;
    kcc_int32                   m_DstXStep              = -1;
    kcc_int32                   m_DstYNum               = -1;
    kcc_int32                   m_DstYStep              = -1;
    kcc_int32                   m_DstZNum               = -1;
    kcc_int32                   m_DstZStep              = -1;
    kcc_int64                   m_DstSbAddress          = -1;
    bool                        m_DstStartAtMidPart     = false;
    DataTypeId                  m_InDtypeId             = DataTypeId::None;
    DataTypeId                  m_BiasDtypeId           = DataTypeId::None;
    kcc_int32                   m_NumPartitions         = -1;
    DataTypeId                  m_OutDtypeId            = DataTypeId::None;
    bool                        m_SrcIsPsum;
    kcc_int32                   m_SrcPsumBankId         = -1;
    kcc_int32                   m_SrcPsumBankOffset     = -1;
    kcc_int32                   m_SrcXNum               = -1;
    kcc_int32                   m_SrcXStep              = -1;
    kcc_int32                   m_SrcYNum               = -1;
    kcc_int32                   m_SrcYStep              = -1;
    kcc_int32                   m_SrcZNum               = 1; // until resolution of SIM
    kcc_int32                   m_SrcZStep              = 1; // //issues.amazon.com/issues/kaena-198
    kcc_int64                   m_SrcSbAddress          = -1;
    bool                        m_SrcStartAtMidPart     = false;
    std::array<kcc_int32, 4>    m_TileId;
    std::string                 m_TileIdFormat          = "";
}; // class ActivationWaveOp::Params : public WaveOp::Params





}}

#endif



