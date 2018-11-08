#pragma once

#ifndef KCC_COMPISA_COMMON_H
#define KCC_COMPISA_COMMON_H

#include <cstring>
#include <typeinfo>

#include "aws_tonga_isa_tpb_common.h"

#include "utils/inc/asserter.hpp"
#include "utils/inc/debug.hpp"

struct TONGA_ISA_TPB_INST_EVENTS;
struct TONGA_ISA_TPB_INST_HEADER;

namespace kcc {
namespace compisa {


using TongaTpbOpcode     = ::TONGA_ISA_TPB_OPCODE;
using TongaErrorCode     = ::TONGA_ISA_ERROR_CODE;
using TongaTpbInstHeader = ::TONGA_ISA_TPB_INST_HEADER;
using TongaTpbEvents     = ::TONGA_ISA_TPB_INST_EVENTS;


template<typename INSTR, TongaTpbOpcode opcode, TongaErrorCode (*Checker)(const INSTR*)>
class InstrTempl : public INSTR {
private:
    using BaseClass = INSTR;
    using Class = InstrTempl;
public:
    InstrTempl() : BaseClass() {
        enum { BYTES_PER_WORD = ::TONGA_ISA_TPB_INST_NBYTES / ::TONGA_ISA_TPB_INST_NWORDS };

        std::memset(this, 0, sizeof(Class)); // zero out instruction
        TongaTpbInstHeader& header(this->inst_header);
        header.opcode = opcode;
        header.inst_word_len = sizeof(BaseClass) / BYTES_PER_WORD;
        header.debug_cmd        = 0;
        header.debug_hint       = 0;
    }

    void CheckValidity() const
    {
        const TongaErrorCode errCode = Checker(this);
        Assert(errCode == ::TONGA_ISA_ERR_CODE_SUCCESS,
               "Invalid instruction of type ", typeid(Class).name(), "    Error code: ", errCode);
    }
};


template<typename INSTR, TongaErrorCode (*Checker)(const INSTR*)>
class InstrTempl2 : public INSTR {
private:
    using BaseClass = INSTR;
    using Class = InstrTempl2;
public:
    InstrTempl2(TongaTpbOpcode opcode) : BaseClass() {
        enum { BYTES_PER_WORD = ::TONGA_ISA_TPB_INST_NBYTES / ::TONGA_ISA_TPB_INST_NWORDS };

        std::memset(this, 0, sizeof(Class)); // zero out instruction
        TongaTpbInstHeader& header(this->inst_header);
        header.opcode = opcode;
        header.inst_word_len = sizeof(Class) / BYTES_PER_WORD;
        header.debug_cmd        = 0;
        header.debug_hint       = 0;
    }

    void CheckValidity() const
    {
        const TongaErrorCode errCode = Checker(this);
        Assert(errCode == ::TONGA_ISA_ERR_CODE_SUCCESS, "Invalid instruction. Error code: ", errCode);
    }
};




void InitSync(TongaTpbEvents& sync);

void InitHeader(TongaTpbInstHeader& header, TongaErrorCode opcode, uint8_t sz);

void InitEvent(uint8_t& event_idx);

template <typename Instr>
void ZeroInstr(Instr& instr)
{
    std::memset(&instr, 0, sizeof(Instr)); // zero out instruction
}

template <typename Instr>
void InitNonEventInstruction(Instr& instr, TongaTpbOpcode opcode)
{
    ZeroInstr(instr);
    InitHeader(instr.inst_header, opcode, sizeof(instr));
}

template <typename Instr>
void InitInstructionWithEmbEvent(Instr &instr, TongaTpbOpcode opcode)
{
    InitNonEventInstruction(instr,opcode);
    //ZeroInstr(instr);
    //InitHeader(instr.inst_header, opcode, sizeof(instr));
    InitSync(instr.inst_events);
}

template <typename Instr>
void InitEventInstruction(Instr& instr, TongaTpbOpcode opcode)
{
    InitNonEventInstruction(instr,opcode);
    //ZeroInstr(instr);
    //InitHeader(instr.inst_header, opcode, sizeof(instr));
    InitEvent(instr.event_idx);
}

}}

#endif // KCC_COMPISA_COMMON_H

