#include "psumbuffer.hpp"

namespace kcc {
namespace arch {

//--------------------------------------------------------
PsumBuffer::PsumBuffer(PeArray* peArray, int numberBanks, int numberBankEntries)
    : m_NumberColumns(peArray->gNumberColumns())
    , m_NumberBanks(numberBanks)
    , m_NumberBankEntries(numberBankEntries)
    , m_BankEntrySizeInBytes(64) // ???
{ }

}}

