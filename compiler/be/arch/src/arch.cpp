#include "poolingeng.hpp"
#include "activationeng.hpp"
#include "statebuffer.hpp"
#include "psumbuffer.hpp"
#include "pearray.hpp"
#include "arch.hpp"

namespace kcc {
namespace arch {

//--------------------------------------------------------
Arch::Arch()
{
    const kcc_int32  numberPeRows            = 128;
    const kcc_int32  numberPeColumns         = 64;
    const kcc_int32  numberPsumBanks         = 4;
    const kcc_int32  numberPsumBankEntries   = 256;

    //const kcc_int64 stateBuffersSizeInBytes = 12 * 1024 * 1024; // numberPeRows  ## 12 MB
    const kcc_int64 stateBuffersSizeInBytes =  8 * 1024 * 1024; // numberPeRows  ##  8 MB

    const kcc_int64 sbPartitionSizeInBytes  = stateBuffersSizeInBytes  / numberPeRows;

    m_PeArray          = new PeArray(numberPeRows, numberPeColumns);  // first

    m_PsumBuffer       = new PsumBuffer(gPeArray(), numberPsumBanks,
                                         numberPsumBankEntries);
    m_PoolingEng       = new PoolingEng(gPsumBuffer());
    m_ActivationEng    = new ActivationEng(gPsumBuffer());
    m_StateBuffer      = new StateBuffer(gPeArray(), sbPartitionSizeInBytes);
}

//----------------------------------------------------------------
kcc_int32 Arch::gNumberPeArrayRows() const
{
    return m_PeArray->gNumberRows();
}

//----------------------------------------------------------------
kcc_int32 Arch::gNumberPeArrayColumns() const
{
    return m_PeArray->gNumberColumns();
}



//----------------------------------------------------------------
kcc_int32 Arch::gNumberPsumBanks() const
{
    return m_PsumBuffer->gNumberBanks();
}

//----------------------------------------------------------------
kcc_int32 Arch::gPsumBankEntries() const
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


