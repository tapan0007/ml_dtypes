
#include "psumbuffer.hpp"
#include "poolingeng.hpp"

namespace kcc {
namespace arch {


PoolingEng::PoolingEng(const PsumBuffer& psumBuffer)
    : m_Width(psumBuffer.gNumberColumns())
{
}

}}

