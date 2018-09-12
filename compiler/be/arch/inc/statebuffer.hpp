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
    static bool qLittleEndian() {
        return true;
    }

    //--------------------------------------------------------
    static bool qTpbReadAccessCheck(TpbAddress address, kcc_uint32 /*size*/) {
        // According to the table, address%4==0, size=4*K, size>=4
        return 0 == (address % 2);
    }

    //--------------------------------------------------------
    static bool qTpbWriteAccessCheck(TpbAddress address, kcc_uint32 /*size*/) {
        // According to the table, address%4==0, size=4*K, size>=4
        return 0 == (address % 4);
    }

    //--------------------------------------------------------
    TpbAddress gEntryTpbAddress(kcc_int32 row, kcc_int32 elmtOffInBytes) const;
    TongaAddress gEntryTongaAddress(kcc_int32 row, kcc_int32 elmtOffInBytes) const;

    TpbAddress gAllZeroOffsetTpbAddress(const utils::DataType& dataType) const;
    TpbAddress gAllOneOffsetTpbAddress(const utils::DataType& dataType) const;

    TongaAddress gAllZeroOffsetTongaAddress(const utils::DataType& dataType) const;
    TongaAddress gAllOneOffsetTongaAddress(const utils::DataType& dataType) const;

private:

    const kcc_int32  m_NumberPartitions;
    const kcc_int64 m_PartitionSizeInBytes;
    const kcc_int64 m_TotalSizeInBytes; // should be last
};

}}

#endif // KCC_ARCH_STATEBUFFER_H
