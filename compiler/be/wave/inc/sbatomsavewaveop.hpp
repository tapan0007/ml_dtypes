
#pragma once

#ifndef KCC_WAVE_SBATOMSAVEWAVEOP_H
#define KCC_WAVE_SBATOMSAVEWAVEOP_H


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


class SbAtomSaveWaveOp : public SbAtomWaveOp {
public:
    class Params;
public:
    SbAtomSaveWaveOp(const SbAtomSaveWaveOp::Params& params, const std::vector<WaveOp*>& prevWaveOps);


    //----------------------------------------------------------------
    bool qSbAtomSaveWaveOp() const override {
        return true;
    }

    static std::string gTypeStr() {
        return WaveOpTypeStr_SBAtomSave;
    }

    bool verify() const override;

    kcc_int32 gOfmapsFoldIdx() const {
        return m_OfmapsFoldIdx;
    }

private:
    kcc_int32       m_OfmapsFoldIdx     = -1;
};




class SbAtomSaveWaveOp::Params : public SbAtomWaveOp::Params {
public:
    bool verify() const;
public:
    kcc_int32       m_OfmapsFoldIdx     = -1;
};

}}


#endif



