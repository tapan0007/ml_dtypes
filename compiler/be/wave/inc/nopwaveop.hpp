
#pragma once

#ifndef KCC_WAVE_NOPWAVEOP_H
#define KCC_WAVE_NOPWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>
#include <array>





#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"

#include "events/inc/events.hpp"

#include "wave/inc/waveop.hpp"


namespace kcc {

namespace wave {


class NopWaveOp : public WaveOp {
private:
    using BaseClass = WaveOp;
public:
    class Params;
public:
    NopWaveOp(const NopWaveOp::Params& params,
              const std::vector<WaveOp*>& prevWaveOps,
              EngineId engineId, events::EventId evt);
public:
    bool verify() const override;

private:
    NopWaveOp() = delete;

public:

    bool qNopWaveOp() const override {
        return true;
    }

    EngineId gEngineId() const override {
        return m_EngineId;
    }

    static std::string gTypeStrStatic();
    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    WaveOpType gType() const override {
        return WaveOpType::Nop;
    }

    const std::string& gLayerName() const override;

    kcc_int32 gReadEventLead() const override {
        return 0;
    }
    kcc_int32 gWriteEventLead() const override {
        return 0;
    }

private:
    EngineId m_EngineId = EngineId::None;
}; // class NopWaveOp : public WaveOp






class NopWaveOp::Params : public WaveOp::Params {
public:
    bool verify() const;
};


}}

#endif


