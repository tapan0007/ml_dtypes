#pragma once

#ifndef KCC_WAVE_TENSORWAVEOP_H
#define KCC_WAVE_TENSORWAVEOP_H


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


class TensorWaveOp : public PoolEngWaveOp {
private:
    using BaseClass = PoolEngWaveOp;
public:
    class Params;
public:
    TensorWaveOp(const TensorWaveOp::Params& params, const std::vector<WaveOp*>& prevWaveOps);
public:
    bool verify() const override;

private:
    TensorWaveOp() = delete;

public:
    static std::string gTypeStrScaleAddStatic() {
        return "ScaleAdd";
    }
    static std::string gTypeStrResAddStatic() {
        return "ResAdd";
    }
    static std::string gTypeStrMultiplyStatic() {
        return "Multiply";
    }
    static std::string gTypeStrSubStatic() {
        return "Sub";
    }
    static std::string gTypeStrAddStatic() {
        return "Add";
    }
    static std::string gTypeStrMaximumStatic() {
        return "Maximum";
    }
    static std::string gTypeStrMinimumStatic() {
        return "Minimum";
    }
    static std::string gTypeStrTensorTensorStatic() {
        return "TensorTensor";
    }
    static std::string gTypeStrTensorScalarStatic() {
        return "TensorScalar";
    }


    std::string gTypeStr() const override;

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

protected:
    std::string     m_TypeStr  = ""; // from JSON

    /* 3 dimensions for src/dst to support batching.  If this instruction runs
     * over sizeof() limit, cut the z dimension! */

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
}; // class TensorWaveOp : public PoolEngWaveOp




class TensorWaveOp::Params : public PoolEngWaveOp::Params {
public:
    bool verify() const;
protected:
public:
    std::string     m_WaveOpType;

    /* 3 dimensions for src/dst to support batching.  If this instruction runs
     * over sizeof() limit, cut the z dimension! */

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
};





}}

#endif




