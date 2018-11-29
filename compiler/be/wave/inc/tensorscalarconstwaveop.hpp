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


#include "wave/inc/tensorwaveop.hpp"


namespace kcc {
namespace wave {


class TensorScalarConstWaveOp : public TensorWaveOp {
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

    virtual WaveOpType gType() const override {
        return WaveOpType::TensorScalarConst;
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



    TensorAluOpType gAluOp(kcc_int32 i) const {
        return m_AluOp[i];
    }
    kcc_float32 gImmVal(kcc_int32 i) const {
        return m_ImmVal[i];
    }

private:
    TensorAluOpType m_AluOp[2];   // operation in Pool ALU
    kcc_float32     m_ImmVal[2];
    
    const DataType& m_InDtype;

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
}; // class TensorScalarConstWaveOp : public PoolEngWaveOp




class TensorScalarConstWaveOp::Params : public TensorWaveOp::Params {
public:
    bool verify() const;
public:
    DataTypeId      m_InDtypeId            = DataTypeId::None;

    kcc_float32     m_ImmVal[2];
    TensorAluOpType m_AluOp[2];

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
};





}}

#endif



