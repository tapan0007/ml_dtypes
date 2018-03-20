#pragma once

#ifndef KCC_ARCH_POOLINGENG_H
#define KCC_ARCH_POOLINGENG_H 1

#include "utils/inc/types.hpp"
#include "arch/inc/archeng.hpp"

namespace kcc {
namespace arch {
class PsumBuffer;
class Arch;

class PoolingEng : public ArchEng {
public:
    PoolingEng(const PsumBuffer& psumBuffer, const Arch& arch);

    kcc_int32 gWidth() const {
        return m_Width;
    }

private:
    const kcc_int32 m_Width;
};

}}


#endif // KCC_ARCH_POOLINGENG_H

