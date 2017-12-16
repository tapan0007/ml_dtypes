#pragma once

#ifndef KCC_ARCH_PSUMBUFFER_H
#define KCC_ARCH_PSUMBUFFER_H 1


namespace kcc {
namespace arch {

class PeArray;

//--------------------------------------------------------
class PsumBuffer {
public:
    //----------------------------------------------------------------
    PsumBuffer(const PeArray* peArray, int numberBanks, int numberBankEntries);

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
    const int  m_NumberColumns;
    const int  m_NumberBanks;
    const int  m_NumberBankEntries;
    const long m_BankEntrySizeInBytes;
};


}}

#endif // KCC_ARCH_PSUMBUFFER_H

