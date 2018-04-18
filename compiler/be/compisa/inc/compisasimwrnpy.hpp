#pragma once

#ifndef KCC_COMPISA_SIM_WRNPY_H
#define KCC_COMPISA_SIM_WRNPY_H



#include "aws_tonga_isa_tpb_sim_wrnpy.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class SimWrNpyInstr : public TONGA_ISA_TPB_SIM_WRNPY_INST {
public:
    static constexpr EngineId engineId = EngineId::DmaEng;
public:
    //----------------------------------------------------------------
    SimWrNpyInstr()
        : TONGA_ISA_TPB_SIM_WRNPY_INST()
    {
        InitSync(inst_events);
    }

};


}}

#endif // KCC_COMPISA_SIM_WRNPY_H

