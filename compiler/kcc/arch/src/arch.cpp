#include "poolingeng.hpp"
#include "activationeng.hpp"
#include "statebuffer.hpp"

#include "arch.hpp"

namespace kcc {
namespace arch {

//--------------------------------------------------------
Arch::Arch()
{
    const int  numberPeRows            = 128;
    const int  numberPeColumns       = 64;
    const int  numberPsumBanks         = 4;
    const int  numberPsumBankEntries   = 256;

    //const long stateBuffersSizeInBytes = 12 * 1024 * 1024; // numberPeRows  ## 12 MB
    const long stateBuffersSizeInBytes =  8 * 1024 * 1024; // numberPeRows  ##  8 MB

    const long sbPartitionSizeInBytes  = stateBuffersSizeInBytes  / numberPeRows;

    m_PeArray          = new PeArray(numberPeRows, numberPeColumns);  // first

    m_PsumBuffer       = new PsumBuffer(gPeArray(), numberPsumBanks,
                                         numberPsumBankEntries);
    m_PoolingEng       = new PoolingEng(gPsumBuffer());
    m_ActivationEng    = new ActivationEng(gPsumBuffer());
    m_StateBuffer      = new StateBuffer(gPeArray(), sbPartitionSizeInBytes);
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

//----------------------------------------------------------------
const std::string&
Arch::gArchVersion() const
{
    static const std::string version("Tonga-0.2");
    return version;
}

}}


