#pragma once

#ifndef KCC_ARCH_ARCH_H
#define KCC_ARCH_ARCH_H 1

#include <string>

#include "types.hpp"

#include "pearray.hpp"
#include "psumbuffer.hpp"
#include "poolingeng.hpp"
#include "activationeng.hpp"
#include "statebuffer.hpp"

namespace kcc {
namespace arch {


//--------------------------------------------------------
class Arch {
private:
    // This is the only row/column configuration for float16 and int16 data types
    // For int8 it is possible to configure pe-array as nrow=256,ncol=64 or as nrow=128,ncol=128
    const kcc_int32  numberPeRows            = 128;
    const kcc_int32  numberPeColumns         = 64;

    const kcc_int32  numberPsumBanks         = 4;
    const kcc_int32  numberPsumBankEntries   = 256;
    const kcc_int64  stateBuffersSizeInBytes = 12 * 1024 * 1024; // numberPeRows  ##  12 MB
    const kcc_int64  sbPartitionSizeInBytes  = stateBuffersSizeInBytes  / numberPeRows;
public:

    //----------------------------------------------------------------
    Arch();

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

private:
    PeArray        m_PeArray;
    PsumBuffer     m_PsumBuffer;
    PoolingEng     m_PoolingEng;
    ActivationEng  m_ActivationEng;
    StateBuffer    m_StateBuffer;
};

}}


#endif // KCC_ARCH_ARCH_H


