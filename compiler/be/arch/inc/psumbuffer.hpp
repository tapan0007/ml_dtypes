#pragma once

#ifndef KCC_ARCH_PSUMBUFFER_H
#define KCC_ARCH_PSUMBUFFER_H 1


#include "utils/inc/types.hpp"

namespace kcc {

namespace utils {
class DataType;
enum class DataTypeId;
}

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
    kcc_int32 gNumberBankEntries(utils::DataTypeId psumDtype) const;

    //----------------------------------------------------------------
    kcc_int32 gNumberColumns() const {
        return m_NumberColumns;
    }

    TpbAddress gEntryTpbAddress(kcc_int32 bankId, kcc_int32 bankEntryIdx, const utils::DataType& dtype) const;

private:
    TpbAddress gPsumBaseTpbAddress() const;
    TongaAddress gBankOffsetDelta() const;
    kcc_int32 gEntrySize () const;
    static bool qLegalDataType(const utils::DataType& dtype);

    //----------------------------------------------------------------
    //TongaAddress gAddress() const;

private:
    const kcc_int32  m_NumberColumns;
    const kcc_int32  m_NumberBanks;
    const kcc_int32  m_NumberBankEntries;
    const kcc_int64  m_BankEntrySizeInBytes;
};


}}

#endif // KCC_ARCH_PSUMBUFFER_H

