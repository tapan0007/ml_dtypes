#include "uarch_cfg.hpp"
#include "shared/inc/tpb_isa.hpp"

#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"
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
PsumBuffer::gPsumBaseAddress() const
{
    return MMAP_PSUM_BASE;
}

kcc_int64
PsumBuffer::gBankOffsetDelta() const
{
    return utils::power2(COLUMN_BANK_SIZE_BITS);
}

kcc_int32
PsumBuffer::gEntrySize () const
{
    return utils::power2(PSUM_ENTRY_BITS);
}

bool
PsumBuffer::qLegalDataType(const utils::DataType& dtype)
{
    switch (dtype.gDataTypeId()) {
    case utils::DataTypeId_Float32:
    case utils::DataTypeId_Int32:
    case utils::DataTypeId_Int64:
        return true;
        break;
    default:
        return false;
        break;
    }
    return false;
}

kcc_int64
PsumBuffer::gEntryTpbAddress(kcc_int32 bankId, kcc_int32 bankEntryIdx,
                             const utils::DataType& dtype) const
{
    Assert(qLegalDataType(dtype), "Wrong data type for PSUM TPB address calculation: ", dtype.gName());
    return gPsumBaseAddress() 
           + gBankOffsetDelta() * bankId 
           + dtype.gSizeInBytes() * bankEntryIdx;
}

}}

