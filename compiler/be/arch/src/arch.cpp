#include "arch/inc/poolingeng.hpp"
#include "arch/inc/activationeng.hpp"
#include "arch/inc/statebuffer.hpp"
#include "arch/inc/psumbuffer.hpp"
#include "arch/inc/pearray.hpp"
#include "arch/inc/arch.hpp"

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
const Arch&
Arch::gArch()
{
    return *s_GlobalArch;
}

void
Arch::init()
{
    s_GlobalArch = std::make_unique<Arch>();
}

std::unique_ptr<Arch> Arch::s_GlobalArch;

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


