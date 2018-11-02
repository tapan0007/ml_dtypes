#pragma once

#include "aws_tonga_isa_tpb_sim_dma_copy.h"
#include "compisa/inc/compisacommon.hpp"


namespace kcc {
namespace compisa {

using SimDmaCopyInstr = InstrTempl<::TONGA_ISA_TPB_SIM_DMA_COPY_INST,
                                   ::TONGA_ISA_TPB_OPCODE_SIM_DMA_COPY,
                                   ::tonga_isa_tpb_simdmacopy_check_validity>;


}}

