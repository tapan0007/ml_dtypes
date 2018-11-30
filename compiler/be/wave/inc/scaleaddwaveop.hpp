#pragma once

#ifndef KCC_WAVE_SCALEADDWAVEOP_H
#define KCC_WAVE_SCALEADDWAVEOP_H


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


/* implements S*x + A, where S = scale, x=input, A=add
 * S and A are immediate values in the instruction
 */
class ScaleAddWaveOp : public PoolEngWaveOp {
private:
    using BaseClass = PoolEngWaveOp;
public:
    class Params;
public:
    ScaleAddWaveOp(const ScaleAddWaveOp::Params& params,
                   const std::vector<WaveOp*>& prevWaveOps);
public:
    bool verify() const override;

private:
    ScaleAddWaveOp() = delete;

public:
    kcc_int64 gDstSbAddress () const {
        return m_DstSbAddress;
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
    kcc_int32 gSrcPsumBankId () const {
        return m_SrcPsumBankId;
    }
    kcc_int32 gSrcPsumBankOffset () const {
        return m_SrcPsumBankOffset;
    }
    kcc_int64 gSrcSbAddress () const {
        return m_SrcSbAddress;
    }
    bool qScaleAddWaveOp() const override {
        return true;
    }

    static std::string gTypeStrStatic();

    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    WaveOpType gType() const override {
        return WaveOpType::ScaleAdd;
    }
    kcc_float32 gScale() const {
        return m_Scale;
    }
    kcc_float32 gOffset() const {
        return m_Offset;
    }

    kcc_int32 gReadEventLead() const override {
        return 0;
    }
    kcc_int32 gWriteEventLead() const override {
        return 0;
    }

private:
    kcc_int64                   m_DstSbAddress          = -1;
    kcc_int32                   m_DstXNum               = -1;
    kcc_int32                   m_DstXStep              = -1;
    kcc_int32                   m_DstYNum               = -1;
    kcc_int32                   m_DstYStep              = -1;
    kcc_int32                   m_DstZNum               = -1;
    kcc_int32                   m_DstZStep              = -1;
    bool                        m_SrcIsPsum             = true;
    kcc_int32                   m_SrcPsumBankId         = -1;
    kcc_int32                   m_SrcPsumBankOffset     = -1;
    kcc_int64                   m_SrcSbAddress          = -1;
    kcc_int32                   m_SrcWNum               = -1;
    kcc_int32                   m_SrcWStep              = -1;
    kcc_int32                   m_SrcXNum               = -1;
    kcc_int32                   m_SrcXStep              = -1;
    kcc_int32                   m_SrcYNum               = -1;
    kcc_int32                   m_SrcYStep              = -1;
    kcc_int32                   m_SrcZNum               = -1;
    kcc_int32                   m_SrcZStep              = -1;
    kcc_float32                 m_Scale                 = 0.0;
    kcc_float32                 m_Offset                = 0.0;
}; // class ScaleAddWaveOp : public WaveOp






class ScaleAddWaveOp::Params : public PoolEngWaveOp::Params {
public:
    bool verify() const;
public:
    kcc_int64                   m_DstSbAddress          = -1;
    kcc_int32                   m_DstXNum               = -1;
    kcc_int32                   m_DstXStep              = -1;
    kcc_int32                   m_DstYNum               = -1;
    kcc_int32                   m_DstYStep              = -1;
    kcc_int32                   m_DstZNum               = -1;
    kcc_int32                   m_DstZStep              = -1;
    bool                        m_SrcIsPsum;
    kcc_int32                   m_SrcPsumBankId;
    kcc_int32                   m_SrcPsumBankOffset;
    kcc_int64                   m_SrcSbAddress;
    kcc_int32                   m_SrcWNum;
    kcc_int32                   m_SrcWStep;
    kcc_int32                   m_SrcXNum;
    kcc_int32                   m_SrcXStep;
    kcc_int32                   m_SrcYNum;
    kcc_int32                   m_SrcYStep;
    kcc_int32                   m_SrcZNum;
    kcc_int32                   m_SrcZStep;
    kcc_float32                 m_Scale                 = 0.0;
    kcc_float32                 m_Offset                = 0.0;
};


}}

#endif


