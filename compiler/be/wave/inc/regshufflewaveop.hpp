#pragma once

#ifndef KCC_WAVE_REGSHUFFLEWAVEOP_H
#define KCC_WAVE_REGSHUFFLEWAVEOP_H


#include <string>
#include <vector>
#include <assert.h>
#include <array>


#include "aws_tonga_isa_tpb_reg_shuffle.h"



#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"

#include "wave/inc/waveop.hpp"


namespace kcc {

namespace wave {


class RegShuffleWaveOp : public WaveOp {
private:
    using BaseClass = WaveOp;
    using Class = RegShuffleWaveOp;
public:
    enum : kcc_int32 {
        MaxNumRegs = TONGA_ISA_TPB_REG_SHUFFLE_MAX_NUM_REGS
    };
public:
    class Params;
public:
    RegShuffleWaveOp(const RegShuffleWaveOp::Params& params,
                  const std::vector<WaveOp*>& prevWaveOps);
public:
    bool verify() const ;

private:
    RegShuffleWaveOp() = delete;

public:
    EngineId gEngineId() const override {
        return EngineId::Pooling;
    }
public:
    static kcc_int32 gMaxNumShuffleRegs() {
        return MaxNumRegs;
    }

    kcc_int32 gStartReg() const {
        return m_StartReg;
    }

    kcc_int32 gInSel(kcc_int32 k) const {
        return m_InSel[k];
    }

    bool qRegShuffleWaveOp() const override {
        return true;
    }

    static std::string gTypeStrStatic();
    std::string gTypeStr() const override {
        return gTypeStrStatic();
    }

    virtual WaveOpType gType() const override {
        return WaveOpType::RegShuffle;
    }

private:
    kcc_int32 m_StartReg = -1;
    std::array<kcc_int32, MaxNumRegs> m_InSel;
}; // class RegShuffleWaveOp : public WaveOp






class RegShuffleWaveOp::Params : public RegShuffleWaveOp::BaseClass::Params {
public:
    bool verify() const;
public:
    kcc_int32 m_StartReg = -1;
    std::array<kcc_int32, MaxNumRegs> m_InSel;

};


}}

#endif


