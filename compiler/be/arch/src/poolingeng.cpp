#include "aws_tonga_isa_tpb_common.h"

#include "arch/inc/psumbuffer.hpp"
#include "arch/inc/poolingeng.hpp"

namespace kcc {
namespace arch {


PoolingEng::PoolingEng(const PsumBuffer& psumBuffer, const Arch& arch)
    : ArchEng(arch)
    , m_Width(psumBuffer.gNumberColumns())
{
}

kcc_int32
PoolingEng::gNumChannels() {
    return TONGA_ISA_TPB_POOLING_NUM_CHANNELS;
}

}}

