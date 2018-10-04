#pragma once

#ifndef KCC_WAVE_TENSORSCALARCONSTWAVEOP_H
#define KCC_WAVE_TENSORSCALARCONSTWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>
#include <array>





#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"

//#include "layers/inc/resaddlayer.hpp"

#include "wave/inc/poolengwaveop.hpp"


namespace kcc {

namespace wave {


class TensorScalarConstWaveOp : public PoolEngWaveOp {
public:
    class Params;
public:
    TensorScalarConstWaveOp(const TensorScalarConstWaveOp::Params& params,
                            const std::vector<WaveOp*>& prevWaveOps);
public:
    bool verify() const override;

private:
    TensorScalarConstWaveOp() = delete;

public:
    bool qTensorScalarConstWaveOp() const override {
        return true;
    }

    std::string gTypeStr() const override;
    static std::string gTypeStrScaleAddStatic() {
        return "ScaleAdd";
    }

    virtual WaveOpType gType() const override {
        return WaveOpType::TensorScalarConst;
    }

    kcc_int32 gNumPartitions () const {
        return m_NumPartitions;
    }

    const DataType& gInDtype () const {
        return m_InDtype;
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
    kcc_int32 gSrcPsumBankId() const {
        return m_SrcPsumBankId;
    }
    kcc_int32 gSrcPsumBankOffset() const {
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
    kcc_int64 gDstSbAddress () const {
        return m_DstSbAddress;
    }
    bool gDstStartAtMidPart () const {
        return m_DstStartAtMidPart;
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

    TensorAluOpType gAluOp(kcc_int32 i) const {
        return m_AluOp[i];
    }
    kcc_float32 gImmVal(kcc_int32 i) const {
        return m_ImmVal[i];
    }

private:
    TensorAluOpType m_AluOp[2];   // operation in Pool ALU
    kcc_float32     m_ImmVal[2];
    std::string     m_TypeStr; // from JSON
    
    const DataType& m_InDtype;

    kcc_int32       m_NumPartitions         = -1;

    /* 3 dimensions for src/dst to support batching.  If this instruction runs
     * over sizeof() limit, cut the z dimension! */

    /* src */
    bool            m_SrcIsPsum;
    kcc_int32       m_SrcPsumBankId        = -1;
    kcc_int32       m_SrcPsumBankOffset    = -1;
    kcc_int32       m_SrcSbAddress         = -1;
    bool            m_SrcStartAtMidPart    = false;

    kcc_int32       m_SrcXStep             = -1;
    kcc_int32       m_SrcXNum              = -1;
    kcc_int32       m_SrcYStep             = -1;
    kcc_int32       m_SrcYNum              = -1;
    kcc_int32       m_SrcZStep             = -1;
    kcc_int32       m_SrcZNum              = -1;

    /* dst */
    bool            m_DstIsPsum;
    kcc_int32       m_DstPsumBankId         = -1;
    kcc_int32       m_DstPsumBankOffset     = -1;
    kcc_int64       m_DstSbAddress          = -1;
    bool            m_DstStartAtMidPart     = false;

    kcc_int32       m_DstXStep              = -1;
    kcc_int32       m_DstXNum               = -1;
    kcc_int32       m_DstYStep              = -1;
    kcc_int32       m_DstYNum               = -1;
    kcc_int32       m_DstZStep              = -1;
    kcc_int32       m_DstZNum               = -1;
}; // class TensorScalarConstWaveOp : public PoolEngWaveOp




class TensorScalarConstWaveOp::Params : public PoolEngWaveOp::Params {
public:
    bool verify() const;
public:
    DataTypeId      m_InDtypeId            = DataTypeId::None;

    kcc_int32       m_NumPartitions         = -1;

    /* 3 dimensions for src/dst to support batching.  If this instruction runs
     * over sizeof() limit, cut the z dimension! */

    /* src_a */
    bool            m_SrcIsPsum;
    kcc_int32       m_SrcPsumBankId        = -1;
    kcc_int32       m_SrcPsumBankOffset    = -1;
    kcc_int64       m_SrcSbAddress         = -1;
    bool            m_SrcStartAtMidPart    = false;

    kcc_int32       m_SrcXStep             = -1;
    kcc_int32       m_SrcXNum              = -1;
    kcc_int32       m_SrcYStep             = -1;
    kcc_int32       m_SrcYNum              = -1;
    kcc_int32       m_SrcZStep             = -1;
    kcc_int32       m_SrcZNum              = -1;

    /* dst */
    bool            m_DstIsPsum;
    kcc_int32       m_DstPsumBankId         = -1;
    kcc_int32       m_DstPsumBankOffset     = -1;
    kcc_int64       m_DstSbAddress          = -1;
    bool            m_DstStartAtMidPart     = false;

    kcc_int32       m_DstXStep              = -1;
    kcc_int32       m_DstXNum               = -1;
    kcc_int32       m_DstYStep              = -1;
    kcc_int32       m_DstYNum               = -1;
    kcc_int32       m_DstZStep              = -1;
    kcc_int32       m_DstZNum               = -1;

    kcc_float32     m_ImmVal[2];
    TensorAluOpType m_AluOp[2];

    std::string     m_WaveOpType;
};





}}

#endif



