#pragma once

#ifndef KCC_ARCH_POOLINGENG_H
#define KCC_ARCH_POOLINGENG_H 1


namespace kcc {
namespace arch {
class PsumBuffer;

class PoolingEng {
public:
    PoolingEng(PsumBuffer* psumBuffer);

    int gWidth() const {
        return m_Width;
    }

    long gInstructionRamStartInBytes() const {
        return 0x001E00000L;
    }

    long gInstructionRamEndInBytes() const {
        return 0x001E03FFFL;
    }

private:
    int m_Width;
};

}}


#endif // KCC_ARCH_POOLINGENG_H

