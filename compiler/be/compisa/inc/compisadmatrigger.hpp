
#pragma once

#include "aws_tonga_isa_tpb_pseudo_dma_trigger.h"

#include "compisa/inc/compisacommon.hpp"

namespace kcc {
namespace compisa {

class DmaTriggerInstr : public TONGA_ISA_TPB_PSEUDO_DMA_TRIGGER_INST {
public:
    // For SB interpret fields in a standard way
    // For DMA interpret fields as follows:
    // start_addr   : descriptor_block_id
    // step_elem[0] : descriptor queue id
    // step_elem[1] : end_event_idx

    DmaTriggerInstr()
        : TONGA_ISA_TPB_PSEUDO_DMA_TRIGGER_INST()
    {
        InitNonEventInstruction(*this, TONGA_ISA_TPB_OPCODE_PSEUDO_DMA_TRIGGER);
    }

    void CheckValidity() const
    {
        tonga_isa_tpb_dma_trigger_check_validity(this);
    }
};


}}

