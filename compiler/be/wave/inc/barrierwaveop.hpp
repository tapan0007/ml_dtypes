#pragma once

#ifndef KCC_WAVE_BARRIERWAVEOP_H
#define KCC_WAVE_BARRIERWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>
#include <array>





#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"


#include "wave/inc/waveop.hpp"


namespace kcc {

namespace wave {


class BarrierWaveOp : public WaveOp {
public:
    class Params;
public:
    BarrierWaveOp(const WaveOp::Params& params,
                  const std::vector<WaveOp*>& prevWaveOps,
                  const std::vector<WaveOp*>& succWaveOps,
                  EngineId engineId);
public:
    bool verify() const override;

    const std::string& gLayerName() const override;

private:
    BarrierWaveOp() = delete;

public:
    bool qBarrierWaveOp() const override {
        return true;
    }

    EngineId gEngineId() const override {
        return m_EngineId;
    }

    static std::string gTypeStrStatic();

    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    virtual WaveOpType gType() const override {
        return WaveOpType::Barrier;
    }

private:
    EngineId m_EngineId = EngineId::AnyEng;
}; // class BarrierWaveOp : public WaveOp


class BarrierWaveOp::Params : public WaveOp::Params {
public:
    bool verify()const ;
};


}}

#endif


