#include "shared/inc/uarch_cfg.hpp"

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



kcc_int64
StateBuffer::gEntryTpbAddress(kcc_int32 row, kcc_int32 elmtOffInBytes) const
{
    return MMAP_SB_BASE + row * utils::power2(ROW_SIZE_BITS) + elmtOffInBytes;
}

kcc_int64
StateBuffer::gEntrySysAddress(kcc_int32 row, kcc_int32 elmtOffInBytes) const
{
    kcc_int64 sysAddr = gEntryTpbAddress(row, elmtOffInBytes);
    sysAddr += Arch::gTpbBaseSysAddress();
    return sysAddr;
}



kcc_int64
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

kcc_int64
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




kcc_int64
StateBuffer::gAllZeroOffsetSysAddress(const utils::DataType& dataType) const
{
    return Arch::gTpbBaseSysAddress() + gAllZeroOffsetTpbAddress(dataType);
}

kcc_int64
StateBuffer::gAllOneOffsetSysAddress(const utils::DataType& dataType) const
{
    return Arch::gTpbBaseSysAddress() + gAllOneOffsetTpbAddress(dataType);
}

}}

