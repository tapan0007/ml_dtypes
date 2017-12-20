#include "pearray.hpp"
#include "statebuffer.hpp"

namespace kcc {
namespace arch {


//--------------------------------------------------------
StateBuffer::StateBuffer(const PeArray* peArray, kcc_int64 partitionSizeInBytes)
    : m_NumberPartitions(peArray->gNumberRows())
    , m_PartitionSizeInBytes(partitionSizeInBytes)
    , m_TotalSizeInBytes(m_NumberPartitions * partitionSizeInBytes)
{
    assert(m_NumberPartitions > 0);
    assert((m_NumberPartitions & (m_NumberPartitions - 1)) == 0); //  power of 2
}


//--------------------------------------------------------
kcc_int64
StateBuffer::gPartitionStartAddressInBytes(kcc_int32 partNum) const
{
    assert((partNum >= 0) && (partNum < gNumberPartitions()));
    return gFirstAddressInBytes() + partNum * gPartitionSizeInBytes();
}

}}

