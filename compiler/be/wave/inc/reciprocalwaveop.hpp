#pragma once

#ifndef KCC_WAVE_RECIPROCALWAVEOP_H
#define KCC_WAVE_RECIPROCALWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>
#include <array>





#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"
#include "wave/inc/poolengwaveop.hpp"


namespace kcc {

namespace wave {


class ReciprocalWaveOp : public PoolEngWaveOp {
private:
    using BaseClass = PoolEngWaveOp;
public:
    class Params;
public:
    ReciprocalWaveOp(const ReciprocalWaveOp::Params& params,
                           const std::vector<WaveOp*>& prevWaveOps);
public:
    bool verify() const override;

private:
    ReciprocalWaveOp() = delete;

public:
    const DataType& gInDtype () const {
        return m_InDtype;
    }
    kcc_int64 gDstSbAddress () const {
        return m_DstSbAddress;
    }
    bool gDstStartAtMidPart () const {
        return m_DstStartAtMidPart;
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
    bool qSrcIsPsum() const {
        return m_SrcIsPsum;
    }
    bool qDstIsPsum() const {
        return m_DstIsPsum;
    }
    kcc_int32 gSrcPsumBankId () const {
        return m_SrcPsumBankId;
    }
    kcc_int32 gSrcPsumBankOffset () const {
        return m_SrcPsumBankOffset;
    }
    kcc_int32 gDstPsumBankId () const {
        return m_DstPsumBankId;
    }
    kcc_int32 gDstPsumBankOffset () const {
        return m_DstPsumBankOffset;
    }
    kcc_int64 gSrcSbAddress () const {
        return m_SrcSbAddress;
    }
    bool gSrcStartAtMidPart () const {
        return m_SrcStartAtMidPart;
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
    const std::array<kcc_int32, 4>& gTileId () const {
        return m_TileId;
    }
    const std::string& gTileIdFormat () const {
        return m_TileIdFormat;
    }

    bool qReciprocalWaveOp() const override {
        return true;
    }

    static std::string gTypeStrStatic();
    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    WaveOpType gType() const override {
        return WaveOpType::Pool;
    }

private:
    const DataType&             m_InDtype;
    kcc_int64                   m_DstSbAddress          = -1;
    bool                        m_DstStartAtMidPart     = false;
    kcc_int32                   m_DstXNum               = -1;
    kcc_int32                   m_DstXStep              = -1;
    kcc_int32                   m_DstYNum               = -1;
    kcc_int32                   m_DstYStep              = -1;
    kcc_int32                   m_DstZNum               = -1;
    kcc_int32                   m_DstZStep              = -1;
    kcc_int32                   m_PoolFrequency         = -1;
    PoolType                    m_PoolFunc              = PoolType::None;
    bool                        m_DstIsPsum             = false;
    bool                        m_SrcIsPsum             = true;
    kcc_int32                   m_SrcPsumBankId         = -1;
    kcc_int32                   m_SrcPsumBankOffset     = -1;
    kcc_int32                   m_DstPsumBankId         = -1;
    kcc_int32                   m_DstPsumBankOffset     = -1;
    kcc_int64                   m_SrcSbAddress          = -1;
    bool                        m_SrcStartAtMidPart     = false;
    kcc_int32                   m_SrcWNum               = -1;
    kcc_int32                   m_SrcWStep              = -1;
    kcc_int32                   m_SrcXNum               = -1;
    kcc_int32                   m_SrcXStep              = -1;
    kcc_int32                   m_SrcYNum               = -1;
    kcc_int32                   m_SrcYStep              = -1;
    kcc_int32                   m_SrcZNum               = -1;
    kcc_int32                   m_SrcZStep              = -1;
    std::array<kcc_int32, 4>    m_TileId;
    std::string                 m_TileIdFormat          = "";
}; // class ReciprocalWaveOp : public PoolEngWaveOp






class ReciprocalWaveOp::Params : public ReciprocalWaveOp::BaseClass::Params {
public:
    bool verify() const;
public:
    DataTypeId                  m_InDtypeId            = DataTypeId::None;
    kcc_int64                   m_DstSbAddress          = -1;
    bool                        m_DstStartAtMidPart     = false;
    kcc_int32                   m_DstXNum               = -1;
    kcc_int32                   m_DstXStep              = -1;
    kcc_int32                   m_DstYNum               = -1;
    kcc_int32                   m_DstYStep              = -1;
    kcc_int32                   m_DstZNum               = -1;
    kcc_int32                   m_DstZStep              = -1;
    bool                        m_SrcIsPsum;
    bool                        m_DstIsPsum;
    kcc_int32                   m_SrcPsumBankId = -1;
    kcc_int32                   m_SrcPsumBankOffset = -1;
    kcc_int32                   m_DstPsumBankId = -1;
    kcc_int32                   m_DstPsumBankOffset = -1;
    kcc_int32                   m_SrcSbAddress = -1;

    bool                        m_SrcStartAtMidPart;

    kcc_int32                   m_SrcXNum = -1;
    kcc_int32                   m_SrcXStep = -1;
    kcc_int32                   m_SrcYNum = -1;
    kcc_int32                   m_SrcYStep = -1;
    kcc_int32                   m_SrcZNum = -1;
    kcc_int32                   m_SrcZStep = -1;
    std::array<kcc_int32, 4>    m_TileId;
    std::string                 m_TileIdFormat;

};


}}

#endif


