#include "psumbuffer.hpp"
#include "activationeng.hpp"

namespace kcc {
namespace arch {

ActivationEng::ActivationEng(PsumBuffer* psumBuffer)
    : m_Width(psumBuffer->gNumberColumns())
{
}

}}


