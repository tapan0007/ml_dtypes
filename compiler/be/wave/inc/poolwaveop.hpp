#pragma once

#ifndef KCC_WAVE_POOLWAVEOP_H
#define KCC_WAVE_POOLWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>
#include <array>





#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"

#include "layers/inc/poollayer.hpp"

#include "wave/inc/poolengwaveop.hpp"


namespace kcc {

namespace wave {


class PoolWaveOp : public PoolEngWaveOp {
public:
    class Params;
public:
    PoolWaveOp(const PoolWaveOp::Params& params,
                           const std::vector<WaveOp*>& prevWaveOps);
public:
    bool verify() const override;

private:
    PoolWaveOp() = delete;

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
    kcc_int32 gNumPartitions () const {
        return m_NumPartitions;
    }
    kcc_int32 gPoolFrequency () const {
        return m_PoolFrequency;
    }
    PoolType gPoolFunc () const {
        return m_PoolFunc;
    }
    bool qSrcIsPsum() const {
        return m_SrcIsPsum;
    }
    kcc_int32 gSrcPsumBankId () const {
        return m_SrcPsumBankId;
    }
    kcc_int32 gSrcPsumBankOffset () const {
        return m_SrcPsumBankOffset;
    }
    kcc_int64 gSrcSbAddress () const {
        return m_SrcSbAddress;
    }
    bool gSrcStartAtMidPart () const {
        return m_SrcStartAtMidPart;
    }
    kcc_int32 gSrcWNum () const {
        return m_SrcWNum;
    }
    kcc_int32 gSrcWStep () const {
        return m_SrcWStep;
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

    bool qPoolWaveOp() const override {
        return true;
    }


    static std::string gTypeStrStatic();
    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    virtual WaveOpType gType() const override {
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
    kcc_int32                   m_NumPartitions         = -1;
    kcc_int32                   m_PoolFrequency         = -1;
    PoolType                    m_PoolFunc              = PoolType::None;
    bool                        m_SrcIsPsum             = true;
    kcc_int32                   m_SrcPsumBankId         = -1;
    kcc_int32                   m_SrcPsumBankOffset     = -1;
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
}; // class PoolWaveOp : public PoolEngWaveOp






class PoolWaveOp::Params : public PoolEngWaveOp::Params {
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
    kcc_int32                   m_NumPartitions         = -1;
    kcc_int32                   m_PoolFrequency         = 0.0;
    PoolType                    m_PoolFunc              = PoolType::None;
    bool                        m_SrcIsPsum;
    kcc_int32                   m_SrcPsumBankId;
    kcc_int32                   m_SrcPsumBankOffset;
    kcc_int64                   m_SrcSbAddress;
    bool                        m_SrcStartAtMidPart;
    kcc_int32                   m_SrcWNum;
    kcc_int32                   m_SrcWStep;
    kcc_int32                   m_SrcXNum;
    kcc_int32                   m_SrcXStep;
    kcc_int32                   m_SrcYNum;
    kcc_int32                   m_SrcYStep;
    kcc_int32                   m_SrcZNum;
    kcc_int32                   m_SrcZStep;
    std::array<kcc_int32, 4>    m_TileId;
    std::string                 m_TileIdFormat;

};


}}

#endif


