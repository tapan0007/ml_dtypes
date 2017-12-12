#pragma once

#ifndef KCC_ARCH_ACTIVATIONENG_H
#define KCC_ARCH_ACTIVATIONENG_H 1


namespace kcc {
namespace arch {

class PsumBuffer;

class ActivationEng {
public:
    ActivationEng(PsumBuffer* psumBuffer);

    int gWidth() const {
        return m_Width;
    }

    long gInstructionRamStartInBytes() const {
        return 0x001F00000L;
    }

    long gInstructionRamEndInBytes() const {
        return 0x001F03FFFL;
    }

private:
    const int m_Width;
};


}}

#endif // KCC_ARCH_ACTIVATIONENG_H

