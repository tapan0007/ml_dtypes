#include "statebuffer.hpp"

namespace kcc {
namespace arch {


//--------------------------------------------------------
StateBuffer::StateBuffer(PeArray* peArray, long partitionSizeInBytes)
{
    m_NumberPartitions = peArray->gNumberRows();
    assert(m_NumberPartitions > 0);
    assert((m_NumberPartitions & (m_NumberPartitions - 1)) == 0); //  power of 2
    m_PartitionSizeInBytes = partitionSizeInBytes;
    m_TotalSizeInBytes     = m_NumberPartitions * m_PartitionSizeInBytes;
}

}}

