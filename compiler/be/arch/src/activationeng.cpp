#include "psumbuffer.hpp"
#include "activationeng.hpp"

namespace kcc {
namespace arch {

ActivationEng::ActivationEng(const PsumBuffer& psumBuffer)
    : m_Width(psumBuffer.gNumberColumns())
{
}

}}


