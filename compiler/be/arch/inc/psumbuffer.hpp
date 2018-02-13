#pragma once

#ifndef KCC_ARCH_PSUMBUFFER_H
#define KCC_ARCH_PSUMBUFFER_H 1


#include "utils/inc/types.hpp"

namespace kcc {
namespace arch {

class PeArray;

//--------------------------------------------------------
class PsumBuffer {
public:
    //----------------------------------------------------------------
    PsumBuffer(const PeArray& peArray, kcc_int32 numberBanks, kcc_int32 numberBankEntries);

    //----------------------------------------------------------------
    kcc_int32 gNumberBanks() const {
        return m_NumberBanks;
    }

    //----------------------------------------------------------------
    kcc_int32 gNumberBankEntries() const {
        return m_NumberBankEntries;
    }

    //----------------------------------------------------------------
    kcc_int32 gBankEntrySizeInBytes() const {
        return m_BankEntrySizeInBytes;
    }

    //----------------------------------------------------------------
    kcc_int32 gNumberColumns() const {
        return m_NumberColumns;
    }

    kcc_int64 gEntryTpbAddress(kcc_int32 bankId, kcc_int32 bankEntryIdx) const {
        return gPsumBaseAddress() + gBankOffsetDelta() * bankId + gEntrySize() * bankEntryIdx;
    }

private:
    kcc_int64 gPsumBaseAddress() const;
    kcc_int64 gBankOffsetDelta() const;
    kcc_int32 gEntrySize () const;

    //----------------------------------------------------------------
    kcc_int64 gAddress() const;

private:
    const kcc_int32  m_NumberColumns;
    const kcc_int32  m_NumberBanks;
    const kcc_int32  m_NumberBankEntries;
    const kcc_int64  m_BankEntrySizeInBytes;
};


}}

#endif // KCC_ARCH_PSUMBUFFER_H

