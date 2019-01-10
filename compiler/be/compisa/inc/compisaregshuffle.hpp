#pragma once

#ifndef KCC_COMPISA_REG_SHUFFLE_H
#define KCC_COMPISA_REG_SHUFFLE_H



#include "aws_tonga_isa_tpb_reg_shuffle.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {
namespace compisa {

using RegShuffleInstrBase = InstrTempl<::TONGA_ISA_TPB_REG_SHUFFLE_INST,
                                       ::TONGA_ISA_TPB_OPCODE_REG_SHUFFLE,
                                       ::tonga_isa_tpb_regshuffle_check_validity>;

class RegShuffleInstr : public RegShuffleInstrBase {
private:
    using BaseClass = RegShuffleInstrBase;
    using Class = RegShuffleInstr;
public:
    kcc_int32 gMaxNumShuffleRegs() const {
        return TONGA_ISA_TPB_REG_SHUFFLE_MAX_NUM_REGS;
    }
};

}}

#endif // KCC_COMPISA_REG_SHUFFLE_H


