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

#include "layers/inc/poollayer.hpp"

#include "wave/inc/waveop.hpp"


namespace kcc {

namespace wave {


class BarrierWaveOp : public WaveOp {
public:
    BarrierWaveOp(const WaveOp::Params& params,
                           const std::vector<WaveOp*>& prevWaveOps,
                           const std::vector<WaveOp*>& succWaveOps);
public:
    bool verify() const override;

private:
    BarrierWaveOp() = delete;

public:
    bool qBarrierWaveOp() const override {
        return true;
    }

    EngineId gEngineId() const override {
        return EngineId::AnyEng;
    }

    static std::string gTypeStrStatic() {
        return WaveOpTypeStr_Barrier;
    }
    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    virtual WaveOpType gType() const override {
        return WaveOpType::Barrier;
    }
}; // class BarrierWaveOp : public WaveOp



}}

#endif


