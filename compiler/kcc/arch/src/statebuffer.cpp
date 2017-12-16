#include "pearray.hpp"
#include "statebuffer.hpp"

namespace kcc {
namespace arch {


//--------------------------------------------------------
StateBuffer::StateBuffer(const PeArray* peArray, long partitionSizeInBytes)
    : m_NumberPartitions(peArray->gNumberRows())
    , m_PartitionSizeInBytes(partitionSizeInBytes)
    , m_TotalSizeInBytes(m_NumberPartitions * partitionSizeInBytes)
{
    assert(m_NumberPartitions > 0);
    assert((m_NumberPartitions & (m_NumberPartitions - 1)) == 0); //  power of 2
}


//--------------------------------------------------------
long
StateBuffer::gPartitionStartAddressInBytes(int partNum) const
{
    assert((partNum >= 0) && (partNum < gNumberPartitions()));
    return gFirstAddressInBytes() + partNum * gPartitionSizeInBytes();
}

}}

