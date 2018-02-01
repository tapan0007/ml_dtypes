#pragma once

#ifndef KCC_ARCH_ACTIVATIONENG_H
#define KCC_ARCH_ACTIVATIONENG_H 1


#include "types.hpp"

namespace kcc {
namespace arch {

class PsumBuffer;

class ActivationEng {
public:
    ActivationEng(const PsumBuffer& psumBuffer);

    kcc_int32 gWidth() const {
        return m_Width;
    }

    kcc_int64 gInstructionRamStartInBytes() const {
        return 0x001F00000L;
    }

    kcc_int64 gInstructionRamEndInBytes() const {
        return 0x001F03FFFL;
    }

private:
    const kcc_int32 m_Width;
};


}}

#endif // KCC_ARCH_ACTIVATIONENG_H

