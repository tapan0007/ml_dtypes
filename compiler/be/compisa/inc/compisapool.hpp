#pragma once

#ifndef KCC_COMPISA_POOL_H
#define KCC_COMPISA_POOL_H



#include "aws_tonga_isa_tpb_pool.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {
namespace compisa {

using PoolInstr = InstrTempl<TONGA_ISA_TPB_POOL_INST,
                             TONGA_ISA_TPB_OPCODE_POOL,
                             tonga_isa_tpb_pool_check_validity>;


}}

#endif // KCC_COMPISA_POOL_H

