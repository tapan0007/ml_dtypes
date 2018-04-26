#pragma once

#ifndef KCC_ARCH_ARCH_H
#define KCC_ARCH_ARCH_H 1

#include <string>
#include <memory>

#include "aws_tonga_isa_tpb_common.h"

#include "utils/inc/types.hpp"

#include "arch/inc/pearray.hpp"
#include "arch/inc/psumbuffer.hpp"
#include "arch/inc/poolingeng.hpp"
#include "arch/inc/activationeng.hpp"
#include "arch/inc/statebuffer.hpp"

namespace kcc {
namespace arch {


// This is the only row/column configuration for float16 and int16 data types
// For int8 it is possible to configure pe-array as nrow=256,ncol=64 or as nrow=128,ncol=128
enum : kcc_int32 {
    Arch_NumberPeRows           = TONGA_ISA_TPB_PE_ARRAY_NUM_ROWS,
    Arch_NumberPeColumns        = TONGA_ISA_TPB_PE_ARRAY_NUM_COLS,

    Arch_NumberPsumBanks        = TONGA_ISA_TPB_PSUM_BUF_NUM_BANKS,
    Arch_PsumEntrySize          = 8, // int64, 4 for fp32
    Arch_NumberPsumBankEntries  = TPB_MMAP_PSUM_BUF_PARTITION_ACTIVE_SIZE
                                  / (Arch_PsumEntrySize * TONGA_ISA_TPB_PSUM_BUF_NUM_BANKS),
};

//--------------------------------------------------------
class Arch {
private:
public:

    //----------------------------------------------------------------
    explicit Arch(kcc_int32 number_events);

    static const Arch& gArch();
    static void init(kcc_int32 number_events);

    //----------------------------------------------------------------
    const PeArray& gPeArray() const {
        return m_PeArray;
    }

    //----------------------------------------------------------------
    const StateBuffer& gStateBuffer() const {
        return m_StateBuffer;
    }

    //----------------------------------------------------------------
    const PsumBuffer& gPsumBuffer() const {
        return m_PsumBuffer;
    }

    //----------------------------------------------------------------
    const PoolingEng& gPoolingEng() const {
        return m_PoolingEng;
    }

    //----------------------------------------------------------------
    const ActivationEng& gActivationEng() const {
        return m_ActivationEng;
    }


    //----------------------------------------------------------------
    kcc_int32 gNumberPeArrayRows() const;

    //----------------------------------------------------------------
    kcc_int32 gNumberPeArrayColumns() const;



    //----------------------------------------------------------------
    kcc_int32 gNumberPsumBanks() const;

    //----------------------------------------------------------------
    kcc_int32 gPsumBankEntries() const;

    //----------------------------------------------------------------
    const std::string& gArchVersion() const;

    static kcc_int64 gTpbBaseSysAddress();

    static kcc_int64 gTpbEventBase();
    static kcc_int64 gSpEventBase();

    kcc_int64 gNumberAllTpbEvents() const;

    static kcc_int64 gNumberSpEvents();

private:
    PeArray        m_PeArray;
    PsumBuffer     m_PsumBuffer;
    PoolingEng     m_PoolingEng;
    ActivationEng  m_ActivationEng;
    StateBuffer    m_StateBuffer;
    kcc_int32      m_NumberTpbEvents = -1;

    static std::unique_ptr<Arch> s_GlobalArch;
};

}}


#endif // KCC_ARCH_ARCH_H


