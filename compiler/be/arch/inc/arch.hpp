#pragma once

#ifndef KCC_ARCH_ARCH_H
#define KCC_ARCH_ARCH_H 1

#include <string>
#include <memory>

#include "utils/inc/types.hpp"

#include "arch/inc/pearray.hpp"
#include "arch/inc/psumbuffer.hpp"
#include "arch/inc/poolingeng.hpp"
#include "arch/inc/activationeng.hpp"
#include "arch/inc/statebuffer.hpp"

namespace kcc {
namespace arch {


//--------------------------------------------------------
class Arch {
private:
public:

    //----------------------------------------------------------------
    explicit Arch();

    static const Arch& gArch();
    static void init();

    //----------------------------------------------------------------
    const PeArray& gPeArray() const {
        return m_PeArray;
    }

    //----------------------------------------------------------------
    const StateBuffer& gStateBuffer() const {
        return m_StateBuffer;
    }

    //----------------------------------------------------------------
    const PsumBuffer& gPsumBuffer() const {
        return m_PsumBuffer;
    }

    //----------------------------------------------------------------
    const PoolingEng& gPoolingEng() const {
        return m_PoolingEng;
    }

    //----------------------------------------------------------------
    const ActivationEng& gActivationEng() const {
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

    static kcc_int64 gTpbBase();

    static kcc_int64 gTbpEventBase();
    static kcc_int64 gSpEventBase();

private:
    PeArray        m_PeArray;
    PsumBuffer     m_PsumBuffer;
    PoolingEng     m_PoolingEng;
    ActivationEng  m_ActivationEng;
    StateBuffer    m_StateBuffer;

    static std::unique_ptr<Arch> s_GlobalArch;
};

}}


#endif // KCC_ARCH_ARCH_H


