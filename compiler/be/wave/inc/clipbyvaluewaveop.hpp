#pragma once

#ifndef KCC_WAVE_CLIPBYVALUEWAVEOP_H
#define KCC_WAVE_CLIPBYVALUEWAVEOP_H


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


class ClipByValueWaveOp : public WaveOp {
public:
    class Params;

public:
    ClipByValueWaveOp(const ClipByValueWaveOp::Params& params,
        const std::vector<WaveOp*>& prevWaveOps);


    bool qSrcIsPsum () const {
        return m_SrcIsPsum;
    }

    kcc_int32 gSrcSbAddress() const {
        return m_SrcSbAddress;
    }
    bool gSrcStartAtMidPart () const {
        return m_SrcStartAtMidPart;
    }

    kcc_int16 gSrcPsumBankId() const {
        return m_SrcPsumBankId;
    }
    kcc_int16 gSrcPsumBankOffset () const {
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


    bool qDstIsPsum () const {
        return m_DstIsPsum;
    }

    kcc_int32 gDstSbAddress() const {
        return m_DstSbAddress;
    }
    bool gDstStartAtMidPart () const {
        return m_DstStartAtMidPart;
    }

    kcc_int16 gDstPsumBankId() const {
        return m_DstPsumBankId;
    }
    kcc_int16 gDstPsumBankOffset () const {
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

    const DataType& gOutDtype () const {
        return m_OutDtype;
    }

    kcc_int32 gNumPartitions () const {
        return m_NumPartitions;
    }

    kcc_float32 gMinValue() const {
        return m_MinValue;
    }
    kcc_float32 gMaxValue() const {
        return m_MaxValue;
    }


    EngineId gEngineId() const override {
        return EngineId::Pooling;
    }

    static std::string gTypeStrStatic();

    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    virtual WaveOpType gType() const override {
        return WaveOpType::ClipByValue;
    }

    const std::string& gTileIdFormat () const {
        return m_TileIdFormat;
    }
    const std::array<kcc_int32, 4>& gTileId () const {
        return m_TileId;
    }

    bool verify() const override;


private:
    ClipByValueWaveOp() = delete;
    ClipByValueWaveOp(const ClipByValueWaveOp&) = delete;

    ClipByValueWaveOp& operator= (const ClipByValueWaveOp&) const = delete;

private:
    const DataType&             m_InDtype;
    const DataType&             m_OutDtype;
    bool                        m_SrcIsPsum             = false;
    bool                        m_DstIsPsum             = false;

    kcc_int32                   m_SrcPsumBankId         = -1;
    kcc_int32                   m_SrcPsumBankOffset     = -1;
    kcc_int32                   m_SrcXNum               = -1;
    kcc_int32                   m_SrcXStep              = -1;
    kcc_int32                   m_SrcYNum               = -1;
    kcc_int32                   m_SrcYStep              = -1;
    kcc_int32                   m_SrcZNum               = -1;
    kcc_int32                   m_SrcZStep              = -1;
    kcc_int64                   m_SrcSbAddress          = -1;
    bool                        m_SrcStartAtMidPart     = false;

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

    kcc_int32                   m_NumPartitions         = -1;
    kcc_float32                 m_MinValue;
    kcc_float32                 m_MaxValue;

    std::array<kcc_int32, 4>    m_TileId;
    std::string                 m_TileIdFormat          = "";
}; // class ClipByValueWaveOp : public WaveOp




class ClipByValueWaveOp::Params : public WaveOp::Params {
public:
    bool verify() const;
public:
    DataTypeId                  m_InDtypeId             = DataTypeId::None;
    DataTypeId                  m_OutDtypeId            = DataTypeId::None;
    bool                        m_SrcIsPsum             = false;
    bool                        m_DstIsPsum             = false;

    kcc_int32                   m_SrcPsumBankId         = -1;
    kcc_int32                   m_SrcPsumBankOffset     = -1;
    kcc_int32                   m_SrcXNum               = -1;
    kcc_int32                   m_SrcXStep              = -1;
    kcc_int32                   m_SrcYNum               = -1;
    kcc_int32                   m_SrcYStep              = -1;
    kcc_int32                   m_SrcZNum               = -1;
    kcc_int32                   m_SrcZStep              = -1;
    kcc_int64                   m_SrcSbAddress          = -1;
    bool                        m_SrcStartAtMidPart     = false;

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

    kcc_int32                   m_NumPartitions         = -1;
    kcc_float32                 m_MinValue;
    kcc_float32                 m_MaxValue;
    std::array<kcc_int32, 4>    m_TileId;
    std::string                 m_TileIdFormat          = "";
};

}}


#endif

