#pragma once

#ifndef KCC_WAVE_RESADDWAVEOP_H
#define KCC_WAVE_RESADDWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>
#include <array>





#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"

#include "layers/inc/resaddlayer.hpp"

#include "wave/inc/poolengwaveop.hpp"


namespace kcc {

namespace wave {


class ResAddWaveOp : public PoolEngWaveOp {
public:
    class Params;
public:
    ResAddWaveOp(const ResAddWaveOp::Params& params,
                           const std::vector<WaveOp*>& prevWaveOps);
public:
    bool verify() const override;

private:
    ResAddWaveOp() = delete;

public:
    bool qResAddWaveOp() const override {
        return true;
    }
    static std::string gTypeStrStatic() {
        return WaveOpTypeStr_ResAdd;
    }
    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    virtual WaveOpType gType() const override {
        return WaveOpType::ResAdd;
    }

    kcc_int32 gNumPartitions () const {
        return m_NumPartitions;
    }

    /* Hack in ResAdd to get Multiply to work with old ISA */
    bool gMultiply () const {
        return m_Multiply;
    }

    // SrcA
    const DataType& gInADtype () const {
        return m_InADtype;
    }
    bool qSrcAIsPsum () const {
        return m_SrcAIsPsum;
    }
    kcc_int64 gSrcASbAddress () const {
        return m_SrcASbAddress;
    }
    kcc_int32 gSrcAPsumBankId() const {
        return m_SrcAPsumBankId;
    }
    kcc_int32 gSrcAPsumBankOffset() const {
        return m_SrcAPsumBankOffset;
    }
    kcc_int32 gSrcAXNum () const {
        return m_SrcAXNum;
    }
    kcc_int32 gSrcAXStep () const {
        return m_SrcAXStep;
    }
    kcc_int32 gSrcAYNum () const {
        return m_SrcAYNum;
    }
    kcc_int32 gSrcAYStep () const {
        return m_SrcAYStep;
    }
    kcc_int32 gSrcAZNum () const {
        return m_SrcAZNum;
    }
    kcc_int32 gSrcAZStep () const {
        return m_SrcAZStep;
    }



    // SrcB
    const DataType& gInBDtype () const {
        return m_InBDtype;
    }
    bool qSrcBIsPsum () const {
        return m_SrcBIsPsum;
    }
    kcc_int64 gSrcBSbAddress () const {
        return m_SrcBSbAddress;
    }
    kcc_int32 gSrcBPsumBankId() const {
        return m_SrcBPsumBankId;
    }
    kcc_int32 gSrcBPsumBankOffset() const {
        return m_SrcBPsumBankOffset;
    }
    kcc_int32 gSrcBXNum () const {
        return m_SrcBXNum;
    }
    kcc_int32 gSrcBXStep () const {
        return m_SrcBXStep;
    }
    kcc_int32 gSrcBYNum () const {
        return m_SrcBYNum;
    }
    kcc_int32 gSrcBYStep () const {
        return m_SrcBYStep;
    }
    kcc_int32 gSrcBZNum () const {
        return m_SrcBZNum;
    }
    kcc_int32 gSrcBZStep () const {
        return m_SrcBZStep;
    }


    bool qDstIsPsum () const {
        return m_DstIsPsum;
    }
    kcc_int64 gDstSbAddress () const {
        return m_DstSbAddress;
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

private:
    const DataType& m_InADtype;
    const DataType& m_InBDtype;

    kcc_int32       m_NumPartitions         = -1;
    bool            m_Multiply;     /* Hack in ResAdd to get Multiply to work with old ISA */

    /* 3 dimensions for src/dst to support batching.  If this instruction runs
     * over sizeof() limit, cut the z dimension! */

    /* src_a */
    bool            m_SrcAIsPsum;
    kcc_int32       m_SrcAPsumBankId        = -1;
    kcc_int32       m_SrcAPsumBankOffset    = -1;
    kcc_int32       m_SrcASbAddress         = -1;

    kcc_int32       m_SrcAXStep             = -1;
    kcc_int32       m_SrcAXNum              = -1;
    kcc_int32       m_SrcAYStep             = -1;
    kcc_int32       m_SrcAYNum              = -1;
    kcc_int32       m_SrcAZStep             = -1;
    kcc_int32       m_SrcAZNum              = -1;

    /* src_b */
    bool            m_SrcBIsPsum;
    kcc_int32       m_SrcBPsumBankId        = -1;
    kcc_int32       m_SrcBPsumBankOffset    = -1;
    kcc_int64       m_SrcBSbAddress         = -1;

    kcc_int32       m_SrcBXStep             = -1;
    kcc_int32       m_SrcBXNum              = -1;
    kcc_int32       m_SrcBYStep             = -1;
    kcc_int32       m_SrcBYNum              = -1;
    kcc_int32       m_SrcBZStep             = -1;
    kcc_int32       m_SrcBZNum              = -1;

    /* dst */
    bool            m_DstIsPsum;
    kcc_int32       m_DstPsumBankId         = -1;
    kcc_int32       m_DstPsumBankOffset     = -1;
    kcc_int64       m_DstSbAddress          = -1;

    kcc_int32       m_DstXStep              = -1;
    kcc_int32       m_DstXNum               = -1;
    kcc_int32       m_DstYStep              = -1;
    kcc_int32       m_DstYNum               = -1;
    kcc_int32       m_DstZStep              = -1;
    kcc_int32       m_DstZNum               = -1;
}; // class ResAddWaveOp : public PoolEngWaveOp




class ResAddWaveOp::Params : public PoolEngWaveOp::Params {
public:
    bool verify() const;
public:
    DataTypeId      m_InADtypeId            = DataTypeId::None;
    DataTypeId      m_InBDtypeId            = DataTypeId::None;

    kcc_int32       m_NumPartitions         = -1;
    bool            m_Multiply;     /* Hack in ResAdd to get Multiply to work with old ISA */

    /* 3 dimensions for src/dst to support batching.  If this instruction runs
     * over sizeof() limit, cut the z dimension! */

    /* src_a */
    bool            m_SrcAIsPsum;
    kcc_int32       m_SrcAPsumBankId        = -1;
    kcc_int32       m_SrcAPsumBankOffset    = -1;
    kcc_int64       m_SrcASbAddress         = -1;

    kcc_int32       m_SrcAXStep             = -1;
    kcc_int32       m_SrcAXNum              = -1;
    kcc_int32       m_SrcAYStep             = -1;
    kcc_int32       m_SrcAYNum              = -1;
    kcc_int32       m_SrcAZStep             = -1;
    kcc_int32       m_SrcAZNum              = -1;

    /* src_b */
    bool            m_SrcBIsPsum;
    kcc_int32       m_SrcBPsumBankId        = -1;
    kcc_int32       m_SrcBPsumBankOffset    = -1;
    kcc_int64       m_SrcBSbAddress         = -1;

    kcc_int32       m_SrcBXStep             = -1;
    kcc_int32       m_SrcBXNum              = -1;
    kcc_int32       m_SrcBYStep             = -1;
    kcc_int32       m_SrcBYNum              = -1;
    kcc_int32       m_SrcBZStep             = -1;
    kcc_int32       m_SrcBZNum              = -1;

    /* dst */
    bool            m_DstIsPsum;
    kcc_int32       m_DstPsumBankId         = -1;
    kcc_int32       m_DstPsumBankOffset     = -1;
    kcc_int64       m_DstSbAddress          = -1;

    kcc_int32       m_DstXStep              = -1;
    kcc_int32       m_DstXNum               = -1;
    kcc_int32       m_DstYStep              = -1;
    kcc_int32       m_DstYNum               = -1;
    kcc_int32       m_DstZStep              = -1;
    kcc_int32       m_DstZNum               = -1;
};





}}

#endif



