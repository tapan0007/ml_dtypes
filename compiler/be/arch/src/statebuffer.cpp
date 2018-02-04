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


//--------------------------------------------------------
kcc_int64
StateBuffer::gPartitionStartAddressInBytes(kcc_int32 partNum) const
{
    assert((partNum >= 0) && (partNum < gNumberPartitions()) && "Partition index not in range");
    return gFirstAddressInBytes() + partNum * gPartitionSizeInBytes();
}

}}

