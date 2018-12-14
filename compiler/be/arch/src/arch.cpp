#include "tonga/address_map.h"
#include "tonga/tpb_addr_map.h"
#include "tonga/sp_addr_map.h"

#include "utils/inc/types.hpp"
#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"

#include "arch/inc/poolingeng.hpp"
#include "arch/inc/activationeng.hpp"
#include "arch/inc/statebuffer.hpp"
#include "arch/inc/psumbuffer.hpp"
#include "arch/inc/pearray.hpp"
#include "arch/inc/arch.hpp"

namespace kcc {
namespace arch {

static_assert(Arch_NumberPeRows == 128, "Number PE rows not 128"); // temporary
static_assert(Arch_NumberPeColumns == 64, "Number PE columns not 64"); // temporary
static_assert(Arch_NumberPsumBanks == 4, "Number PSUM banks not 4"); // temporary
static_assert(Arch_NumberPsumBankEntries == 256, "Number PSUM entries not 256"); // temporary

enum : kcc_int64 {
    sbPartitionSizeInBytes  = TONGA_ISA_TPB_STATE_BUF_PARTITION_ACTIVE_SIZE,
    stateBuffersSizeInBytes = sbPartitionSizeInBytes  * Arch_NumberPeRows
};

//--------------------------------------------------------
Arch::Arch(kcc_int32 numTpbEvents)
    : m_PeArray(Arch_NumberPeRows, Arch_NumberPeColumns, *this)
    , m_PsumBuffer(m_PeArray, Arch_NumberPsumBanks, Arch_NumberPsumBankEntries)
    , m_PoolingEng(m_PsumBuffer, *this)
    , m_ActivationEng(m_PsumBuffer, *this)
    , m_StateBuffer(m_PeArray, sbPartitionSizeInBytes)
{
    enum {
        // TONGA_ISA_TPB_NUM_EVENTS is in arch-isa repo in file tpb/aws_tonga_isa_tpb_common.h
        // MMAP_TPB_TPB_EVT_SZ is in arch-headers repo in files rel_addr_map.h and tpb_addr_map.h
        // The following should hold:
        // MMAP_TPB_TPB_EVT_SZ = TONGA_ISA_TPB_NUM_EVENTS * TONGA_ISA_TPB_EVENT_SIZE
        NumTpbEvents = TONGA_ISA_TPB_NUM_EVENTS  // this is from isa/tpb
    };

    static_assert(NumTpbEvents <=
        (1U << 8*sizeof(TONGA_ISA_TPB_INST_EVENTS::wait_event_idx)),
        "Number of TPB events too large for type Event_t");

    m_NumberTpbEvents = numTpbEvents > 0 ? numTpbEvents : NumTpbEvents;
}

//----------------------------------------------------------------
const Arch&
Arch::gArch()
{
    return *s_GlobalArch;
}

void
Arch::init(kcc_int32 number_events)
{
    s_GlobalArch = std::make_unique<Arch>(number_events);
}

std::unique_ptr<Arch> Arch::s_GlobalArch;

//----------------------------------------------------------------
kcc_int32 Arch::gNumberPeArrayRows() const
{
    return m_PeArray.gNumberRows();
}

//----------------------------------------------------------------
kcc_int32 Arch::gNumberPeArrayColumns() const
{
    return m_PeArray.gNumberColumns();
}



//----------------------------------------------------------------
kcc_int32 Arch::gNumberPsumBanks() const
{
    return m_PsumBuffer.gNumberBanks();
}

//----------------------------------------------------------------
kcc_int32 Arch::gPsumBankEntries(utils::DataTypeId psumDtype) const
{
    return m_PsumBuffer.gNumberBankEntries(psumDtype);
}

//----------------------------------------------------------------
const std::string&
Arch::gArchVersion() const
{
    static const std::string version("Tonga-0.2");
    return version;
}

TongaAddress
Arch::gTpbEventBase()
{
    return P_0_TPB_0_TPB_EVT_RELBASE;

}

TongaAddress
Arch::gSpEventBase()
{
    return P_0_SP_0_EVENT_RELBASE;
}

TongaAddress
Arch::gTpbBaseTongaAddress()
{
    return P_0_TPB_0_RELBASE;
}

TpbAddress
Arch::gTpbAddressOfStateBuffer()
{
    return P_0_TPB_0_STATE_BUF_RELBASE;
}

kcc_int64
Arch::gNumberSpEvents()
{
    return MMAP_SP_EVENT_SZ;
}

kcc_int64
Arch::gNumberAllTpbEvents() const
{
    Assert(m_NumberTpbEvents > 0,
        "Number of events must be positive: ", m_NumberTpbEvents);
    return m_NumberTpbEvents;
}


}}



