#pragma once

#ifndef KCC_COMPISA_COMMON_H
#define KCC_COMPISA_COMMON_H

#include <cstring>

#include "aws_tonga_isa_tpb_common.h"

#include "utils/inc/asserter.hpp"

struct TONGA_ISA_TPB_INST_EVENTS;
struct TONGA_ISA_TPB_INST_HEADER;

namespace kcc {
namespace compisa {


using TongaTpbOpcode = TONGA_ISA_TPB_OPCODE;


template<typename INSTR, TongaTpbOpcode opcode, TONGA_ISA_ERROR_CODE (*Checker)(const INSTR*)>
class InstrTempl : public INSTR {
public:
    InstrTempl() : INSTR() {
        enum { BYTES_PER_WORD = TONGA_ISA_TPB_INST_NBYTES / TONGA_ISA_TPB_INST_NWORDS };

        std::memset(this, 0, sizeof(InstrTempl)); // zero out instruction
        TONGA_ISA_TPB_INST_HEADER& header(this->inst_header);
        header.opcode = opcode;
        header.inst_word_len = sizeof(INSTR) / BYTES_PER_WORD;
        header.debug_cmd        = 0;
        header.debug_hint       = 0;
    }

    void CheckValidity() const
    {
        const TONGA_ISA_ERROR_CODE errCode = Checker(this);
        Assert(errCode == TONGA_ISA_ERR_CODE_SUCCESS, "Invalid instruction");
    }
};



void InitSync(TONGA_ISA_TPB_INST_EVENTS& sync);

void InitHeader(TONGA_ISA_TPB_INST_HEADER& header, TONGA_ISA_TPB_OPCODE opcode, uint8_t sz);

void InitEvent(uint8_t& event_idx);

template <typename Instr>
void ZeroInstr(Instr& instr)
{
    std::memset(&instr, 0, sizeof(Instr)); // zero out instruction
}

template <typename Instr>
void InitNonEventInstruction(Instr& instr, TONGA_ISA_TPB_OPCODE opcode)
{
    ZeroInstr(instr);
    InitHeader(instr.inst_header, opcode, sizeof(instr));
}

template <typename Instr>
void InitInstructionWithEmbEvent(Instr &instr, TONGA_ISA_TPB_OPCODE opcode)
{
    InitNonEventInstruction(instr,opcode);
    //ZeroInstr(instr);
    //InitHeader(instr.inst_header, opcode, sizeof(instr));
    InitSync(instr.inst_events);
}

template <typename Instr>
void InitEventInstruction(Instr& instr, TONGA_ISA_TPB_OPCODE opcode)
{
    InitNonEventInstruction(instr,opcode);
    //ZeroInstr(instr);
    //InitHeader(instr.inst_header, opcode, sizeof(instr));
    InitEvent(instr.event_idx);
}

}}

#endif // KCC_COMPISA_COMMON_H

