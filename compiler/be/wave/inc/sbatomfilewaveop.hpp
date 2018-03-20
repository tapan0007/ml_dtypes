#pragma once

#ifndef KCC_WAVE_SBATOMFILEWAVEOP_H
#define KCC_WAVE_SBATOMFILEWAVEOP_H


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


class SbAtomFileWaveOp : public SbAtomWaveOp {
public:
    class Params;
public:
    SbAtomFileWaveOp(const SbAtomFileWaveOp::Params& params, const std::vector<WaveOp*>& prevWaveOps);


    //----------------------------------------------------------------
    bool qSbAtomFileWaveOp() const override {
        return true;
    }

    static std::string gTypeStrStatic() {
        return WaveOpTypeStr_SBAtomFile;
    }
    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    bool verify() const override;

    kcc_int32 gIfmapsFoldIdx() const {
        return m_IfmapsFoldIdx;
    }

    bool qIfmapsReplicate() const {
        return m_IfmapsReplicate;
    }

    bool qContainWeights() const {
        return m_ContainWeights;
    }

    kcc_int32 gIfmapCount () const {
        return m_IfmapCount;
    }

    kcc_int64 gLoadDataSizeInBytes () const;

private:
    kcc_int32       m_IfmapCount        = -1;
    kcc_int32       m_IfmapsFoldIdx     = -1;
    bool            m_IfmapsReplicate   = false;
    bool            m_ContainWeights    = false;
};

using SbAtomLoadWaveOp = SbAtomFileWaveOp;




class SbAtomFileWaveOp::Params : public SbAtomWaveOp::Params {
public:
    bool verify() const;
public:
    kcc_int32       m_IfmapCount        = -1;
    kcc_int32       m_IfmapsFoldIdx     = -1;
    bool            m_IfmapsReplicate   = false;
    bool            m_ContainWeights    = false;
};

}}


#endif




