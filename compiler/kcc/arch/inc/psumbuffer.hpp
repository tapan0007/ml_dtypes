#pragma once

#ifndef KCC_ARCH_PSUMBUFFER_H
#define KCC_ARCH_PSUMBUFFER_H 1

#include "pearray.hpp"

namespace kcc {
namespace arch {

//--------------------------------------------------------
class PsumBuffer {
public:
    //----------------------------------------------------------------
    PsumBuffer(PeArray* peArray, int numberBanks, int numberBankEntries);

    //----------------------------------------------------------------
    int gNumberBanks() const {
        return m_NumberBanks;
    }

    //----------------------------------------------------------------
    int gNumberBankEntries() const {
        return m_NumberBankEntries;
    }

    //----------------------------------------------------------------
    int gBankEntrySizeInBytes() const {
        return m_BankEntrySizeInBytes;
    }

    //----------------------------------------------------------------
    int gNumberColumns() const {
        return m_NumberColumns;
    }

    //----------------------------------------------------------------
    long gAddress() const {
        return 0x001800000L;
    }

private:
    int m_NumberColumns;
    int m_NumberBanks;
    int m_NumberBankEntries;
    long m_BankEntrySizeInBytes;
};


}}

#endif // KCC_ARCH_PSUMBUFFER_H

