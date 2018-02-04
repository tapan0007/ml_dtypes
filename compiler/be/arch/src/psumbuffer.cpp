
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

}}

