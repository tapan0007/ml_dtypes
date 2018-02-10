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

    static std::string gTypeStr() {
        return WaveOpTypeStr_SBAtomFile;
    }

    bool verify() const override;

    kcc_int32 gIfmapsFoldIdx() const {
        return m_IfmapsFoldIdx;
    }

    bool qIfmapsReplicate() const {
        return m_IfmapsReplicate;
    }

private:
    kcc_int32       m_IfmapsFoldIdx     = -1;
    bool            m_IfmapsReplicate   = -1;
};




class SbAtomFileWaveOp::Params : public SbAtomWaveOp::Params {
public:
    bool verify() const;
public:
    kcc_int32       m_IfmapsFoldIdx     = -1;
    bool            m_IfmapsReplicate   = true;
};

}}


#endif



