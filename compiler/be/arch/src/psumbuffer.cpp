#include "tonga/address_map.h"

#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/types.hpp"

#include "arch/inc/arch.hpp"
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
TpbAddress
PsumBuffer::gPsumBaseTpbAddress() const
{
    return P_0_TPB_0_PSUM_BUF_RELBASE;
}

/*
Ron:
Ok, I didn't answer well before (had a mistake in the gaps description).

Take #2:
The relevant section from the TPB spec is copied below.
So each partition is 16KB, and it's arranged as follows:
* 0KB-2KB:  bank0 data
* 2KB-4KB:  bank1 data
* 4KB-6KB:  bank2 data
* 6KB-8KB:  bank3 data
* 8KB-10KB:  bank0 metadata
  @ We keep a bit per entry to indicate if the result overflowed or not
  @ This is mostly important in integer networks (you overflow and saturate to 255 ? but you don?t know if this is a real 255 or a fake 255, etc?.)
* 10KB-12KB:  bank1 metadata
* ...

So... Banks are indeed un-padded.
But, my answer to the 2nd email was incorrect -- address arrangement is bank0 - bank1 - bank2 - bank3 - 8KB 'padding'
(it's not really padding, but from sw perspective it is)


Sorry about the confusion, let me know if anything is not clear...

From the TPB spec:

16KB allocated per-partition. Internal memory allocation:
* Partition-Index: ADDR[19:14] are used to indicate the relevant partition index.
  @ Legal values:  0 - 63
  @ Illegal values:  N/A
* Data/Metadata:  ADDR[13] is used to indicate access to partial-sum (0), or to overflow metadata indications (1).
* Bank-Select: ADDR[12:11] are used to select bank inside the partition
  @ Legal values:  0 - 3
  @ Illegal values:  N/A
* Bank-Psum-Offset:  ADDR[10:3] are used to indicate offset within the bank.
Byte-Offset:  ADDR[2:0] are used to indicate byte-offset the bank-entry.

From ISA tpb_common.h

TONGA_ISA_TPB_PSUM_BUF_NUM_PARTITIONS         =  64,
TONGA_ISA_TPB_PSUM_BUF_NUM_BANKS              =   4,
TONGA_ISA_TPB_PSUM_BUF_PARTITION_SIZE         = ( 16 * 1024), // PSUM partitions are 16KB apart
TONGA_ISA_TPB_PSUM_BUF_PARTITION_ACTIVE_SIZE  = (  8 * 1024),
TONGA_ISA_TPB_PSUM_BUF_BANK_SIZE     = (TONGA_ISA_TPB_PSUM_BUF_PARTITION_ACTIVE_SIZE/TONGA_ISA_TPB_PSUM_BUF_NUM_BANKS),
*/

TongaAddress
PsumBuffer::gBankOffsetDelta() const
{
    return TONGA_ISA_TPB_PSUM_BUF_BANK_SIZE;
}

kcc_int32
PsumBuffer::gEntrySize() const
{
    return Arch_PsumEntrySize;
}

bool
PsumBuffer::qLegalDataType(const utils::DataType& dtype)
{
    switch (dtype.gDataTypeId()) {
    case utils::DataTypeId::Float32:
    case utils::DataTypeId::Int32:
    case utils::DataTypeId::Int64:
        return true;
        break;
    default:
        return false;
        break;
    }
    return false;
}

TpbAddress
PsumBuffer::gEntryTpbAddress(kcc_int32 bankId, kcc_int32 bankEntryIdx,
                             const utils::DataType& dtype) const
{
    Assert(qLegalDataType(dtype), "Wrong data type for PSUM TPB address calculation: ", dtype.gName());
    TpbAddress addr = gPsumBaseTpbAddress();
    addr += gBankOffsetDelta() * bankId;
    addr += dtype.gSizeInBytes() * bankEntryIdx;
    return addr;
}

}}

