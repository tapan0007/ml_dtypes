#include "arch/inc/psumbuffer.hpp"
#include "arch/inc/activationeng.hpp"

namespace kcc {
namespace arch {

ActivationEng::ActivationEng(const PsumBuffer& psumBuffer, const Arch& arch)
    : ArchEng(arch)
    , m_Width(psumBuffer.gNumberColumns())
{
}

}}


