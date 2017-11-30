#include "poolingeng.hpp"
#include "activationeng.hpp"
#include "statebuffer.hpp"

#include "arch.hpp"

namespace kcc {
namespace arch {

//--------------------------------------------------------
Arch::Arch()
{
    m_NumberPeRows            = 128;
    m_NumberPeColumns         = 64;
    m_NumberPsumBanks         = 4;
    m_NumberPsumBankEntries   = 256;
    m_SbPartitionsSize        = 12 * 1024 * 1024; // numberPeRows  ## 12 MB
    m_SbPartitionsSize        =  8 * 1024 * 1024; // numberPeRows  ##  8 MB

    m_PeArray          = new PeArray(m_NumberPeRows, m_NumberPeColumns);
    m_PsumBuffer       = new PsumBuffer(gPeArray(), m_NumberPsumBanks,
                                         m_NumberPsumBankEntries);
    m_PoolingEng       = new PoolingEng(gPsumBuffer());
    m_ActivationEng    = new ActivationEng(gPsumBuffer());
    m_StateBuffer      = new StateBuffer(gPeArray(), m_SbPartitionsSize);
}

//----------------------------------------------------------------
int Arch::gNumberPeArrayRows() const
{
    return m_PeArray->gNumberRows();
}

//----------------------------------------------------------------
int Arch::gNumberPeArrayColumns() const
{
    return m_PeArray->gNumberColumns();
}



//----------------------------------------------------------------
int Arch::gNumberPsumBanks() const
{
    return m_PsumBuffer->gNumberBanks();
}

//----------------------------------------------------------------
int Arch::gPsumBankEntries() const
{
    return m_PsumBuffer->gNumberBankEntries();
}

}}


