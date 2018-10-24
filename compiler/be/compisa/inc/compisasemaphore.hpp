#pragma once

#ifndef KCC_COMPSEMAPHORE_NOP_H
#define KCC_COMPSEMAPHORE_NOP_H



#include "aws_tonga_isa_tpb_semaphore_test_and_set.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {

using SemaphoreInstr = InstrTempl<TONGA_ISA_TPB_SEMAPHORE_TEST_AND_SET_INST,
                                  TONGA_ISA_TPB_OPCODE_SEMAPHORE_TEST_AND_SET,
                                  tonga_isa_tpb_semaphore_test_and_set_check_validity>;


}}

#endif // KCC_COMPSEMAPHORE_NOP_H


