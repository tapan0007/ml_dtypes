
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
private:
    using BaseClass = SbAtomWaveOp;
public:
    class Params;
public:
    SbAtomSaveWaveOp(const SbAtomSaveWaveOp::Params& params, const std::vector<WaveOp*>& prevWaveOps);


    //----------------------------------------------------------------
    bool qSbAtomSaveWaveOp() const override {
        return true;
    }

    static std::string gTypeStrStatic();

    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    WaveOpType gType() const override {
        return WaveOpType::Save;
    }

    bool verify() const override;


    kcc_int64 gSaveDataSizeInBytes() const;

    bool qFinalLayerOfmap() const {
        return m_FinalLayerOfmap;
    }

    kcc_int32 gReadEventLead() const override {
        return 0;
    }
    kcc_int32 gWriteEventLead() const override {
        return 0;
    }

private:
    bool            m_FinalLayerOfmap   = false;
};




class SbAtomSaveWaveOp::Params : public SbAtomWaveOp::Params {
public:
    bool verify() const;
public:
    bool            m_FinalLayerOfmap   = false;
};

}}


#endif



