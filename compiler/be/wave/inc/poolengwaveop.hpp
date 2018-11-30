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
private:
    using BaseClass = WaveOp;
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
    kcc_int32 gNumPartitions () const {
        return m_NumPartitions;
    }

    kcc_int32 gReadEventLead() const override {
        return gNumPartitions();
    }
    kcc_int32 gWriteEventLead() const override {
        return gNumPartitions();
    }

protected:
    bool verify() const;

protected:
    const DataType&     m_OutDtype;
    kcc_int32           m_NumPartitions = -1;
}; // class PoolEngWaveOp : public WaveOp






class PoolEngWaveOp::Params : public PoolEngWaveOp::BaseClass::Params {
public:
    DataTypeId m_OutDtypeId    = DataTypeId::None;
    kcc_int32  m_NumPartitions = -1;
};


}}

#endif



