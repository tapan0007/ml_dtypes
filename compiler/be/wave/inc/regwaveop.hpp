#pragma once

#ifndef KCC_WAVE_REGWAVEOP_H
#define KCC_WAVE_REGWAVEOP_H


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


class RegWaveOp : public WaveOp {
private:
    using BaseClass = WaveOp;
public:
    class Params;
public:
    RegWaveOp(const RegWaveOp::Params& params,
                  const std::vector<WaveOp*>& prevWaveOps);
public:
    bool verify() const override;

private:
    RegWaveOp() = delete;

public:
    EngineId gEngineId() const override {
        return EngineId::Pooling;
    }
    kcc_int32 gReadEventLead() const override {
        return gNumPartitions();
    }
    kcc_int32 gWriteEventLead() const override {
        return gNumPartitions();
    }

    kcc_int32 gNumPartitions () const {
        return m_NumPartitions;
    }
    bool qParallelMode() const {
        return m_ParallelMode;
    }

private:
    kcc_int32                   m_NumPartitions = -1;
    bool                        m_ParallelMode  = true;
}; // class RegWaveOp : public WaveOp






class RegWaveOp::Params : public RegWaveOp::BaseClass::Params {
public:
    bool verify() const;
public:
    kcc_int32                   m_NumPartitions = -1;
    bool                        m_ParallelMode          = true;
};


}}

#endif



