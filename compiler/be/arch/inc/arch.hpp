#pragma once

#ifndef KCC_ARCH_ARCH_H
#define KCC_ARCH_ARCH_H 1

#include <string>

#include "types.hpp"

namespace kcc {
namespace arch {
class PoolingEng;
class PeArray;
class PsumBuffer;
class ActivationEng;
class StateBuffer;


//--------------------------------------------------------
class Arch {
public:

    //----------------------------------------------------------------
    Arch();

    //----------------------------------------------------------------
    PeArray* gPeArray() {
        return m_PeArray;
    }

    //----------------------------------------------------------------
    StateBuffer* gStateBuffer() {
        return m_StateBuffer;
    }

    //----------------------------------------------------------------
    PsumBuffer* gPsumBuffer() {
        return m_PsumBuffer;
    }

    //----------------------------------------------------------------
    PoolingEng* gPoolingEng() {
        return m_PoolingEng;
    }

    //----------------------------------------------------------------
    ActivationEng* gActivationEng() {
        return m_ActivationEng;
    }


    //----------------------------------------------------------------
    kcc_int32 gNumberPeArrayRows() const;

    //----------------------------------------------------------------
    kcc_int32 gNumberPeArrayColumns() const;



    //----------------------------------------------------------------
    kcc_int32 gNumberPsumBanks() const;

    //----------------------------------------------------------------
    kcc_int32 gPsumBankEntries() const;

    //----------------------------------------------------------------
    const std::string& gArchVersion() const;

private:
    PeArray*       m_PeArray;
    StateBuffer*   m_StateBuffer;
    PsumBuffer*    m_PsumBuffer;
    PoolingEng*    m_PoolingEng;
    ActivationEng* m_ActivationEng;
};

}}


#endif // KCC_ARCH_ARCH_H


