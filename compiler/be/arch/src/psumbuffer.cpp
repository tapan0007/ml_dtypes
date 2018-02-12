#include "uarch_cfg.hpp"
#include "tpb_isa.hpp"

#include "utils/inc/types.hpp"

#include "arch/inc/pearray.hpp"
#include "arch/inc/psumbuffer.hpp"

namespace kcc {
namespace arch {

//--------------------------------------------------------
PsumBuffer::PsumBuffer(const PeArray& peArray, kcc_int32 numberBanks, kcc_int32 numberBankEntries)
    : m_NumberColumns(peArray.gNumberColumns())
    , m_NumberBanks(numberBanks)
    , m_NumberBankEntries(numberBankEntries)
    , m_BankEntrySizeInBytes(64) // ???
{ }


//----------------------------------------------------------------
kcc_int64
PsumBuffer::gAddress() const
{
    return 0x02000000L;
}

kcc_int64
PsumBuffer::gPsumBaseAddress() const
{
    return MMAP_PSUM_BASE;
}

kcc_int64
PsumBuffer::gBankOffsetDelta() const
{
    return power2(COLUMN_BANK_SIZE_BITS);
}

kcc_int32
PsumBuffer::gEntrySize () const
{
    return power2(PSUM_ENTRY_BITS);
}

}}

