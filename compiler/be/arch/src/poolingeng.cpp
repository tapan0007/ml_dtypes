
#include "arch/inc/psumbuffer.hpp"
#include "arch/inc/poolingeng.hpp"

namespace kcc {
namespace arch {


PoolingEng::PoolingEng(const PsumBuffer& psumBuffer, const Arch& arch)
    : ArchEng(arch)
    , m_Width(psumBuffer.gNumberColumns())
{
}

}}

