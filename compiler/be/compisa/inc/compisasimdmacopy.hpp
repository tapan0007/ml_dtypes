#pragma once

#include "aws_tonga_isa_tpb_sim_dma_copy.h"
#include "compisa/inc/compisacommon.hpp"


namespace kcc {
namespace compisa {

class SimDmaCopyInstr : public TONGA_ISA_TPB_SIM_DMA_COPY_INST {
public:
    SimDmaCopyInstr()
        : TONGA_ISA_TPB_SIM_DMA_COPY_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_SIM_DMA_COPY);
    }

    void CheckValidity() const
    {
        tonga_isa_tpb_simdmacopy_check_validity(this);
    }
};


}}

