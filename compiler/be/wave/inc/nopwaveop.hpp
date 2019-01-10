
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
    enum class NopType {
        Barrier,
        Broadcast,
        None,
    };
public:
    NopWaveOp(const NopWaveOp::Params& params,
              const std::vector<WaveOp*>& prevWaveOps,
              EngineId engineId, events::EventId evt,
              NopType type);
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


    NopType gNopType() const {
        return m_NopType;
    }

    bool qPartOfBarrier() const override {
        return m_NopType == NopType::Barrier;
    }

private:
    EngineId m_EngineId = EngineId::None;
    const NopType m_NopType;
}; // class NopWaveOp : public WaveOp






class NopWaveOp::Params : public WaveOp::Params {
public:
    bool verify() const;
};


}}

#endif


