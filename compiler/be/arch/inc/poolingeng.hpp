#pragma once

#ifndef KCC_ARCH_POOLINGENG_H
#define KCC_ARCH_POOLINGENG_H 1

#include "utils/inc/types.hpp"

namespace kcc {
namespace arch {
class PsumBuffer;

class PoolingEng {
public:
    PoolingEng(const PsumBuffer& psumBuffer);

    kcc_int32 gWidth() const {
        return m_Width;
    }

    kcc_int64 gInstructionRamStartInBytes() const {
        return 0x001E00000L;
    }

    kcc_int64 gInstructionRamEndInBytes() const {
        return 0x001E03FFFL;
    }

private:
    const kcc_int32 m_Width;
};

}}


#endif // KCC_ARCH_POOLINGENG_H

