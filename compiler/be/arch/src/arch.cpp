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
    : m_PeArray(numberPeRows, numberPeColumns)
    , m_PsumBuffer(m_PeArray, numberPsumBanks, numberPsumBankEntries)
    , m_PoolingEng(m_PsumBuffer)
    , m_ActivationEng(m_PsumBuffer)
    , m_StateBuffer(m_PeArray, sbPartitionSizeInBytes)
{
}

//----------------------------------------------------------------
kcc_int32 Arch::gNumberPeArrayRows() const
{
    return m_PeArray.gNumberRows();
}

//----------------------------------------------------------------
kcc_int32 Arch::gNumberPeArrayColumns() const
{
    return m_PeArray.gNumberColumns();
}



//----------------------------------------------------------------
kcc_int32 Arch::gNumberPsumBanks() const
{
    return m_PsumBuffer.gNumberBanks();
}

//----------------------------------------------------------------
kcc_int32 Arch::gPsumBankEntries() const
{
    return m_PsumBuffer.gNumberBankEntries();
}

//----------------------------------------------------------------
const std::string&
Arch::gArchVersion() const
{
    static const std::string version("Tonga-0.2");
    return version;
}

}}


