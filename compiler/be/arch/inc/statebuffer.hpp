#pragma once

#ifndef KCC_ARCH_STATEBUFFER_H
#define KCC_ARCH_STATEBUFFER_H 1


#include "utils/inc/types.hpp"

namespace kcc {
namespace utils {
class DataType;
}

namespace arch {

class PeArray;

//--------------------------------------------------------
class StateBuffer {
public:
    //--------------------------------------------------------
    StateBuffer(const PeArray& peArray, kcc_int64 partitionSizeInBytes);

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
    kcc_int64 gEntryTpbAddress(kcc_int32 row, kcc_int32 elmtOffInBytes) const;
    kcc_int64 gEntrySysAddress(kcc_int32 row, kcc_int32 elmtOffInBytes) const;

    kcc_int64 gAllZeroOffsetTpbAddress(const utils::DataType& dataType) const;
    kcc_int64 gAllOneOffsetTpbAddress(const utils::DataType& dataType) const;

private:

    const kcc_int32  m_NumberPartitions;
    const kcc_int64 m_PartitionSizeInBytes;
    const kcc_int64 m_TotalSizeInBytes; // should be last
};

}}

#endif // KCC_ARCH_STATEBUFFER_H
