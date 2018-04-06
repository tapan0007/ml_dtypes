
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

    static std::string gTypeStrStatic() {
        return WaveOpTypeStr_Nop;
    }
    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    virtual WaveOpType gType() const override {
        return WaveOpType::Nop;
    }

    const std::string& gLayerName() const override;

private:
    EngineId m_EngineId = EngineId::AnyEng;
}; // class NopWaveOp : public WaveOp






class NopWaveOp::Params : public WaveOp::Params {
public:
    bool verify() const;
};


}}

#endif


