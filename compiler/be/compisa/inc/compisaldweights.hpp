#pragma once

#ifndef KCC_COMPISA_LDWEIGHTS_H
#define KCC_COMPISA_LDWEIGHTS_H



#include "aws_tonga_isa_tpb_ldweights.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {

using LdWeightsInstr = InstrTempl<::TONGA_ISA_TPB_LDWEIGHTS_INST,
                                  ::TONGA_ISA_TPB_OPCODE_LDWEIGHTS,
                                  ::tonga_isa_tpb_ldweights_check_validity>;


}}

#endif // KCC_COMPISA_LDWEIGHTS_H

