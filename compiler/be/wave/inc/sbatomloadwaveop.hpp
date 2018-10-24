#pragma once

#ifndef KCC_WAVE_SBATOMLOADWAVEOP_H
#define KCC_WAVE_SBATOMLOADWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>





#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"
#include "wave/inc/waveop.hpp"
#include "wave/inc/sbatomwaveop.hpp"


namespace kcc {
namespace wave {


class SbAtomLoadWaveOp : public SbAtomWaveOp {
public:
    class Params;
public:
    SbAtomLoadWaveOp(const SbAtomLoadWaveOp::Params& params, const std::vector<WaveOp*>& prevWaveOps);


    //----------------------------------------------------------------
    bool qSbAtomLoadWaveOp() const override {
        return true;
    }

    static std::string gTypeStrStatic();

    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    virtual WaveOpType gType() const override {
        return WaveOpType::Load;
    }


    bool verify() const override;

    bool qContainWeights() const {
        return m_ContainWeights;
    }

    kcc_int64 gLoadDataSizeInBytes () const;

    kcc_int32 gIfmapReplicationNumRows() const {
        return m_IfmapReplicationNumRows;
    }
    kcc_int32 gIfmapReplicationResolution() const {
        return m_IfmapReplicationResolution;
    }
    kcc_int32       gIfmapReplicationStepBytes() const {
        return m_IfmapReplicationStepBytes;
    }

    kcc_int32 gSrcStepElem() const {
        return m_SrcStepElem;
    }

private:
    bool            m_ContainWeights    = false;

    kcc_int32       m_IfmapReplicationNumRows       = -1;
    kcc_int32       m_IfmapReplicationResolution    = -1;
    kcc_int32       m_IfmapReplicationStepBytes     = -1;
    bool            m_IfmapsReplicate               = false;

    kcc_int32       m_SrcStepElem                   = -1;
}; // class SbAtomLoadWaveOp : public SbAtomWaveOp





class SbAtomLoadWaveOp::Params : public SbAtomWaveOp::Params {
public:
    bool verify() const;
public:
    bool            m_ContainWeights    = false;

    kcc_int32       m_IfmapReplicationNumRows       = -1;
    kcc_int32       m_IfmapReplicationResolution    = -1;
    kcc_int32       m_IfmapReplicationStepBytes     = -1;
    bool            m_IfmapsReplicate               = false;

    kcc_int32       m_SrcStepElem                   = -1;
}; // class SbAtomLoadWaveOp::Params : public SbAtomWaveOp::Params

}}


#endif




