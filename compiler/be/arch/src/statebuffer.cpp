#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/pearray.hpp"
#include "arch/inc/statebuffer.hpp"

namespace kcc {
namespace arch {


//--------------------------------------------------------
StateBuffer::StateBuffer(const PeArray& peArray, kcc_int64 partitionSizeInBytes)
    : m_NumberPartitions(peArray.gNumberRows())
    , m_PartitionSizeInBytes(partitionSizeInBytes)
    , m_TotalSizeInBytes(m_NumberPartitions * partitionSizeInBytes)
{
    Assert(m_NumberPartitions > 0, "Number of partitions in SB is not positive: ", m_NumberPartitions);
    Assert(( (m_NumberPartitions & (m_NumberPartitions - 1)) == 0), "Number of partitions in SB is not power of 2: ", m_NumberPartitions);
}



TpbAddress
StateBuffer::gEntryTpbAddress(kcc_int32 row, kcc_int32 elmtOffInBytes) const
{
    return Arch::gTpbAddressOfStateBuffer() + row * TONGA_ISA_TPB_STATE_BUF_PARTITION_SIZE + elmtOffInBytes;
}

TongaAddress
StateBuffer::gEntryTongaAddress(kcc_int32 row, kcc_int32 elmtOffInBytes) const
{
    TongaAddress sysAddr = gEntryTpbAddress(row, elmtOffInBytes);
    sysAddr += Arch::gTpbBaseTongaAddress();
    return sysAddr;
}

/* Replace it with proper values. */
#define MMAP_SB_FP32_ZERO_OFFSET       0x18000
#define MMAP_SB_FP32_ONE_OFFSET        0x18100
#define MMAP_SB_INT32_ZERO_OFFSET      0x18200
#define MMAP_SB_INT32_ONE_OFFSET       0x18300

TpbAddress
StateBuffer::gAllZeroOffsetTpbAddress(const utils::DataType& dataType) const
{
    if (dynamic_cast<const utils::DataTypeFloat32*>(&dataType)) {
        return MMAP_SB_FP32_ZERO_OFFSET;
    }
    if (dynamic_cast<const utils::DataTypeFloat16*>(&dataType)) {
        return MMAP_SB_FP32_ZERO_OFFSET;
    }

    if (dynamic_cast<const utils::DataTypeUint8*>(&dataType)) {
        return MMAP_SB_INT32_ZERO_OFFSET;
    }
    if (dynamic_cast<const utils::DataTypeUint16*>(&dataType)) {
        return MMAP_SB_INT32_ZERO_OFFSET;
    }

    if (dynamic_cast<const utils::DataTypeInt32*>(&dataType)) {
        return MMAP_SB_INT32_ZERO_OFFSET;
    }
    if (dynamic_cast<const utils::DataTypeInt64*>(&dataType)) {
        return MMAP_SB_INT32_ZERO_OFFSET;
    }
    Assert(false, "No all-one in State Buffer for data type ", dataType.gName());
    return 0;
}

TpbAddress
StateBuffer::gAllOneOffsetTpbAddress(const utils::DataType& dataType) const
{
    if (dynamic_cast<const utils::DataTypeFloat32*>(&dataType)) {
        return MMAP_SB_FP32_ONE_OFFSET;
    }
    if (dynamic_cast<const utils::DataTypeFloat16*>(&dataType)) {
        Assert(false, "Float16 does not have 1.0 in State Buffer");
        return 0;
    }
    if (dynamic_cast<const utils::DataTypeUint8*>(&dataType)) {
        return MMAP_SB_INT32_ONE_OFFSET;
    }
    if (dynamic_cast<const utils::DataTypeUint16*>(&dataType)) {
        return MMAP_SB_INT32_ONE_OFFSET;
    }

    if (dynamic_cast<const utils::DataTypeInt32*>(&dataType)) {
        return MMAP_SB_INT32_ONE_OFFSET;
    }
    if (dynamic_cast<const utils::DataTypeInt64*>(&dataType)) {
        return MMAP_SB_INT32_ONE_OFFSET;
    }
    Assert(false, "No all-one in State Buffer for data type ", dataType.gName());
    return 0;
}




TongaAddress
StateBuffer::gAllZeroOffsetTongaAddress(const utils::DataType& dataType) const
{
    return Arch::gTpbBaseTongaAddress() + gAllZeroOffsetTpbAddress(dataType);
}

TongaAddress
StateBuffer::gAllOneOffsetTongaAddress(const utils::DataType& dataType) const
{
    return Arch::gTpbBaseTongaAddress() + gAllOneOffsetTpbAddress(dataType);
}

}}

