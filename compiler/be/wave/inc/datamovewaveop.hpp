#pragma once

#ifndef KCC_WAVE_DATAMOVEWAVEOP_H
#define KCC_WAVE_DATAMOVEWAVEOP_H


#include <string>
#include <array>
#include <assert.h>





#include "utils/inc/debug.hpp"
#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"
#include "wave/inc/waveop.hpp"

namespace kcc {

namespace dma {
class DmaQueue;
}

namespace wave {

class DataMoveWaveOp : public WaveOp {
private:
    using BaseClass = WaveOp;
public:
    class Params;
public:
    DataMoveWaveOp(const DataMoveWaveOp::Params& params, const std::vector<WaveOp*>& prevWaveOps);

    bool qDataMoveWaveOp() const override{
        return true;
    }

    //----------------------------------------------------------------
    EngineId gEngineId() const override {
        return m_EngineId;
    }
    void rEngineId(EngineId engId) {
        m_EngineId = engId;
    }

    const dma::DmaQueue* gDmaQueue() const {
        return m_DmaQueue;
    }
    void rDmaQueue(const dma::DmaQueue* dmaQueue) {
        m_DmaQueue = dmaQueue;
    }

    kcc_int32 gTriggerOrd() const {
        return m_TriggerOrd;
    }
    void rTriggerOrd(kcc_int32 ord) {
        m_TriggerOrd = ord;
    }


protected:
    bool verify() const override;

private:
    EngineId                m_EngineId      = EngineId::None;
    const dma::DmaQueue*    m_DmaQueue      = nullptr;
    kcc_int32               m_TriggerOrd    = -1;
};




class DataMoveWaveOp::Params : public BaseClass::Params {
public:
    bool verify() const;
};

}}


#endif



