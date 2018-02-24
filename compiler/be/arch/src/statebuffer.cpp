#include "uarch_cfg.hpp"

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
    assert(m_NumberPartitions > 0 && "Number of partitions in SB not positive");
    assert(( (m_NumberPartitions & (m_NumberPartitions - 1)) == 0) && "Number of partitions not power of 2");
}



kcc_int64
StateBuffer::gEntryTpbAddress(kcc_int32 row, kcc_int32 elmtOffInBytes) const
{
    return MMAP_SB_BASE + row * utils::power2(ROW_SIZE_BITS) + elmtOffInBytes;
}

kcc_int64
StateBuffer::gAllZeroOffsetTpbAddress() const
{
    return MMAP_SB_FP32_ZERO_OFFSET;
}

}}

