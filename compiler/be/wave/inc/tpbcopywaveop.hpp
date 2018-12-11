#pragma once

#ifndef KCC_WAVE_COPYWAVEOP_H
#define KCC_WAVE_COPYWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>





#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"
#include "wave/inc/datamovewaveop.hpp"


namespace kcc {
namespace wave {

class SbAtomLoadWaveOp;


class TpbCopyWaveOp : public DataMoveWaveOp {
private:
    using BaseClass = DataMoveWaveOp;
public:
    class Params;
public:
    TpbCopyWaveOp(const TpbCopyWaveOp::Params& params, const std::vector<WaveOp*>& prevWaveOps);


    //----------------------------------------------------------------
    bool qTpbCopyWaveOp() const override {
        return true;
    }

    static std::string gTypeStrStatic();

    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    WaveOpType gType() const override {
        return WaveOpType::TpbCopy;
    }


    bool verify() const override;

    kcc_int64 gLoadDataSizeInBytes () const;

    kcc_int32 gReadEventLead() const override {
        return 0;
    }
    kcc_int32 gWriteEventLead() const override {
        return 0;
    }

    const SbAtomLoadWaveOp* gPairLoadWaveOp() const {
        return m_PairLoadWaveOp;
    }
    const TpbCopyWaveOp* gPrevCopyWaveOp() const {
        return m_PrevCopyWaveOp;
    }

    TpbAddress gSrcSbAddress() const {
        return m_SrcSbAddress;
    }
    TpbAddress gDstSbAddress() const {
        return m_DstSbAddress;
    }
    kcc_int64 gSizeInBytes() const {
        return m_SizeInBytes;
    }

private:
    SbAtomLoadWaveOp*   m_PairLoadWaveOp    = nullptr;
    TpbCopyWaveOp*      m_PrevCopyWaveOp    = nullptr;
    kcc_int64           m_SrcSbAddress      = -1;
    kcc_int64           m_DstSbAddress      = -1;
    kcc_int64           m_SizeInBytes       = -1;
}; // class TpbCopyWaveOp : public WaveOp





class TpbCopyWaveOp::Params : public BaseClass::Params {
public:
    bool verify() const;
public:
    SbAtomLoadWaveOp*   m_PairLoadWaveOp    = nullptr;
    TpbCopyWaveOp*      m_PrevCopyWaveOp    = nullptr;
    kcc_int64           m_SrcSbAddress      = -1;
    kcc_int64           m_DstSbAddress      = -1;
    kcc_int64           m_SizeInBytes       = -1;
}; // class TpbCopyWaveOp::Params : public SbAtomWaveOp::Params

}}


#endif




