#include "shared/inc/uarch_cfg.hpp"

#include "utils/inc/types.hpp"

#include "arch/inc/poolingeng.hpp"
#include "arch/inc/activationeng.hpp"
#include "arch/inc/statebuffer.hpp"
#include "arch/inc/psumbuffer.hpp"
#include "arch/inc/pearray.hpp"
#include "arch/inc/arch.hpp"

namespace kcc {
namespace arch {

// This is the only row/column configuration for float16 and int16 data types
// For int8 it is possible to configure pe-array as nrow=256,ncol=64 or as nrow=128,ncol=128
enum : kcc_int32 {
    Arch_NumberPeRows            = utils::power2(ROW_BITS),
    Arch_NumberPeColumns         = utils::power2(COLUMN_BITS),

    Arch_NumberPsumBanks         = utils::power2(BANKS_PER_COLUMN_BITS),
    Arch_NumberPsumBankEntries   = utils::power2(PSUM_NUM_ENTRY_BITS),
};

static_assert(Arch_NumberPeRows == 128, "Number PE rows not 128"); // temporary
static_assert(Arch_NumberPeColumns == 64, "Number PE columns not 64"); // temporary
static_assert(Arch_NumberPsumBanks == 4, "Number PSUM banks not 4"); // temporary
static_assert(Arch_NumberPsumBankEntries == 256, "Number PSUM entries not 256"); // temporary

enum : kcc_int64 {
    sbPartitionSizeInBytes  = TPB_MMAP_STATE_BUF_PARTITION_ACTIVE_SIZE,
    stateBuffersSizeInBytes = sbPartitionSizeInBytes  * Arch_NumberPeRows
};

//--------------------------------------------------------
Arch::Arch()
    : m_PeArray(Arch_NumberPeRows, Arch_NumberPeColumns, *this)
    , m_PsumBuffer(m_PeArray, Arch_NumberPsumBanks, Arch_NumberPsumBankEntries)
    , m_PoolingEng(m_PsumBuffer, *this)
    , m_ActivationEng(m_PsumBuffer, *this)
    , m_StateBuffer(m_PeArray, sbPartitionSizeInBytes)
{
}

//----------------------------------------------------------------
const Arch&
Arch::gArch()
{
    return *s_GlobalArch;
}

void
Arch::init()
{
    s_GlobalArch = std::make_unique<Arch>();
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
kcc_int32 Arch::gPsumBankEntries() const
{
    return m_PsumBuffer.gNumberBankEntries();
}

//----------------------------------------------------------------
const std::string&
Arch::gArchVersion() const
{
    static const std::string version("Tonga-0.2");
    return version;
}

kcc_int64
Arch::gTpbEventBase()
{
    return MMAP_EVENTS;

}

kcc_int64
Arch::gSpEventBase()
{
    return SP_EVENTS_BASE;
}

kcc_int64
Arch::gTpbBase()
{
    return TPB_BASE;
}

}}


