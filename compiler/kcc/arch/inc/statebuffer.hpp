#pragma once

#ifndef KCC_ARCH_STATEBUFFER_H
#define KCC_ARCH_STATEBUFFER_H 1

#include "pearray.hpp"

namespace kcc {
namespace arch {

//--------------------------------------------------------
class StateBuffer {
public:
    //--------------------------------------------------------
    StateBuffer(PeArray* peArray, long partitionSizeInBytes);

    //--------------------------------------------------------
    int gNumberPartitions() const {
        return m_NumberPartitions;
    }

    //--------------------------------------------------------
    long gPartitionSizeInBytes() const {
        return m_PartitionSizeInBytes;
    }

    //--------------------------------------------------------
    long gTotalSizeInBytes() const {
        return m_TotalSizeInBytes;
    }

    //--------------------------------------------------------
    bool qLittleEndian() const {
        return true;
    }

    //--------------------------------------------------------
    long gFirstAddressInBytes() const {
        return 0x000000000L;
    }


    //--------------------------------------------------------
    long gPartitionStartAddressInBytes(int partNum) const {
        assert((partNum >= 0) && (partNum < gNumberPartitions()));
        return gFirstAddressInBytes() + partNum * gPartitionSizeInBytes();
    }

private:

    int m_NumberPartitions;
    long m_PartitionSizeInBytes;
    long m_TotalSizeInBytes;
};

}}

#endif // KCC_ARCH_STATEBUFFER_H
