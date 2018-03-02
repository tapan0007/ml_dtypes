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

#include "wave/inc/waveop.hpp"


namespace kcc {

namespace wave {


class PoolWaveOp : public WaveOp {
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
    kcc_int32 gDstSbAtomId () const {
        return m_DstSbAtomId;
    }
    kcc_int32 gDstSbOffsetInAtom () const {
        return m_DstSbOffsetInAtom;
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
    kcc_int32 gNumPartitions () const {
        return m_NumPartitions;
    }
    const DataType& gOutDtype () const {
        return m_OutDtype;
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
    kcc_int32 gSrcSbAtomId () const {
        return m_SrcSbAtomId;
    }
    kcc_int32 gSrcSbOffsetInAtom () const {
        return m_SrcSbOffsetInAtom;
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
    static std::string gTypeStr() {
        return WaveOpTypeStr_Pool;
    }

private:
    kcc_int32                   m_DstSbAtomId          = -1;
    kcc_int32                   m_DstSbOffsetInAtom     = -1;
    kcc_int32                   m_DstXNum               = -1;
    kcc_int32                   m_DstXStep              = -1;
    kcc_int32                   m_DstYNum               = -1;
    kcc_int32                   m_DstYStep              = -1;
    kcc_int32                   m_DstZNum               = -1;
    kcc_int32                   m_DstZStep              = -1;
    const DataType&             m_InDtype;
    // "layername;
    kcc_int32                   m_NumPartitions         = -1;
    const DataType&             m_OutDtype;
    kcc_int32                   m_PoolFrequency         = -1;
    PoolType                    m_PoolFunc              = PoolType_None;
    // previouswaveops;
    //  1conv/i1/MatMuln0m0h0w0c0r0s0"
    // ],
    bool                        m_SrcIsPsum             = true;
    kcc_int32                   m_SrcPsumBankId         = -1;
    kcc_int32                   m_SrcPsumBankOffset     = -1;
    kcc_int32                   m_SrcSbAtomId           = -1;
    kcc_int32                   m_SrcSbOffsetInAtom     = -1;
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
    //waveopname;
    //waveoptype;
}; // class PoolWaveOp : public WaveOp






class PoolWaveOp::Params : public WaveOp::Params {
public:
    bool verify() const;
public:
    kcc_int32                   m_DstSbAtomId           = -1;
    kcc_int32                   m_DstSbOffsetInAtom     = -1;
    kcc_int32                   m_DstXNum               = -1;
    kcc_int32                   m_DstXStep              = -1;
    kcc_int32                   m_DstYNum               = -1;
    kcc_int32                   m_DstYStep              = -1;
    kcc_int32                   m_DstZNum               = -1;
    kcc_int32                   m_DstZStep              = -1;
    DataTypeId                  m_InDtype               = DataTypeId_None;
    // "layername;
    kcc_int32                   m_NumPartitions         = -1;
    DataTypeId                  m_OutDtype              = DataTypeId_None;
    kcc_int32                   m_PoolFrequency         = 0.0;
    PoolType                    m_PoolFunc              = PoolType_None;
    // previouswaveops;
    //  1conv/i1/MatMuln0m0h0w0c0r0s0"
    // ],
    bool                        m_SrcIsPsum;
    kcc_int32                   m_SrcPsumBankId;
    kcc_int32                   m_SrcPsumBankOffset;
    kcc_int32                   m_SrcSbAtomId;
    kcc_int32                   m_SrcSbOffsetInAtom;
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
    //waveopname;
    //waveoptype;

};


}}

#endif


