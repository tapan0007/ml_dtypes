#ifndef KCC_ARCH_ARCH_H
#define KCC_ARCH_ARCH_H 1

#include "psumbuffer.hpp"
#include "pearray.hpp"
//#include "statebuffer.hpp"
//#include "poolingeng.hpp"
//#include "activationeng.hpp"

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
    int gNumberPeArrayRows() const;

    //----------------------------------------------------------------
    int gNumberPeArrayColumns() const;



    //----------------------------------------------------------------
    int gNumberPsumBanks() const;

    //----------------------------------------------------------------
    int gPsumBankEntries() const;

private:
    int m_NumberPeRows;
    int m_NumberPeColumns;
    int m_NumberPsumBanks;
    int m_NumberPsumBankEntries;
    long m_SbPartitionsSize;

    PeArray*       m_PeArray;
    StateBuffer*   m_StateBuffer;
    PsumBuffer*    m_PsumBuffer;
    PoolingEng*    m_PoolingEng;
    ActivationEng* m_ActivationEng;
};

}}


#endif // KCC_ARCH_ARCH_H


