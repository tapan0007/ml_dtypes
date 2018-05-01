#pragma once

#ifndef KCC_COMPISA_COMMON_H
#define KCC_COMPISA_COMMON_H

#include <cstring>

#include "aws_tonga_isa_tpb_common.h"

struct TONGA_ISA_TPB_INST_EVENTS;
struct TONGA_ISA_TPB_INST_HEADER;

namespace kcc {
namespace compisa {

void InitSync(TONGA_ISA_TPB_INST_EVENTS& sync);

void InitHeader(TONGA_ISA_TPB_INST_HEADER& header, TONGA_ISA_TPB_OPCODE opcode, uint8_t sz);

void InitEvent(uint8_t& event_idx);

template <typename Instr>
void ZeroInstr(Instr& instr)
{
    std::memset(&instr, 0, sizeof(Instr)); // zero out instruction
}

template <typename Instr>
void InitInstructionWithEmbEvent(Instr &instr, TONGA_ISA_TPB_OPCODE opcode)
{
    ZeroInstr(instr);
    InitHeader(instr.inst_header, opcode, sizeof(instr));
    InitSync(instr.inst_events);
}

template <typename Instr>
void InitEventInstruction(Instr& instr, TONGA_ISA_TPB_OPCODE opcode)
{
    ZeroInstr(instr);
    InitHeader(instr.inst_header, opcode, sizeof(instr));
    InitEvent(instr.event_idx);
}

}}

#endif // KCC_COMPISA_COMMON_H

