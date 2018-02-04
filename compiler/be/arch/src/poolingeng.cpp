
#include "arch/inc/psumbuffer.hpp"
#include "arch/inc/poolingeng.hpp"

namespace kcc {
namespace arch {


PoolingEng::PoolingEng(const PsumBuffer& psumBuffer)
    : m_Width(psumBuffer.gNumberColumns())
{
}

}}

