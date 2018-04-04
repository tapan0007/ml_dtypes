#pragma once

#ifndef KCC_WAVE_POOLENGWAVEOP_H
#define KCC_WAVE_POOLENGWAVEOP_H


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


class PoolEngWaveOp : public WaveOp {
public:
    class Params;
public:
    PoolEngWaveOp(const PoolEngWaveOp::Params& params,
                  const std::vector<WaveOp*>& prevWaveOps);

private:
    PoolEngWaveOp() = delete;

public:
    const DataType& gOutDtype () const {
        return m_OutDtype;
    }
    EngineId gEngineId() const override {
        return EngineId::Pooling;
    }

private:
    const DataType&             m_OutDtype;
}; // class PoolEngWaveOp : public WaveOp






class PoolEngWaveOp::Params : public WaveOp::Params {
public:
    DataTypeId m_OutDtypeId = DataTypeId::None;
};


}}

#endif



