#pragma once

#ifndef KCC_COMPISA_REG_STORE_H
#define KCC_COMPISA_REG_STORE_H



#include "aws_tonga_isa_tpb_reg_store.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {
namespace compisa {

using RegStoreInstr = InstrTempl<TONGA_ISA_TPB_REG_STORE_INST,
                             TONGA_ISA_TPB_OPCODE_REG_STORE,
                             tonga_isa_tpb_regstore_check_validity>;


}}

#endif // KCC_COMPISA_REG_STORE_H


