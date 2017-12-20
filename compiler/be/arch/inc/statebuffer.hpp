#pragma once

#ifndef KCC_ARCH_STATEBUFFER_H
#define KCC_ARCH_STATEBUFFER_H 1

#include "types.hpp"

namespace kcc {
namespace arch {

class PeArray;

//--------------------------------------------------------
class StateBuffer {
public:
    //--------------------------------------------------------
    StateBuffer(const PeArray* peArray, kcc_int64 partitionSizeInBytes);

    //--------------------------------------------------------
    kcc_int32 gNumberPartitions() const {
        return m_NumberPartitions;
    }

    //--------------------------------------------------------
    kcc_int64 gPartitionSizeInBytes() const {
        return m_PartitionSizeInBytes;
    }

    //--------------------------------------------------------
    kcc_int64 gTotalSizeInBytes() const {
        return m_TotalSizeInBytes;
    }

    //--------------------------------------------------------
    bool qLittleEndian() const {
        return true;
    }

    //--------------------------------------------------------
    kcc_int64 gFirstAddressInBytes() const {
        return 0x000000000L;
    }


    //--------------------------------------------------------
    kcc_int64 gPartitionStartAddressInBytes(kcc_int32 partNum) const;

private:

    const kcc_int32  m_NumberPartitions;
    const kcc_int64 m_PartitionSizeInBytes;
    const kcc_int64 m_TotalSizeInBytes; // should be last
};

}}

#endif // KCC_ARCH_STATEBUFFER_H
