#pragma once

#ifndef KCC_ARCH_ARCHENG_H
#define KCC_ARCH_ARCHENG_H 1


#include "utils/inc/types.hpp"
//#include "arch/inc/arch.hpp"

namespace kcc {
namespace arch {

class Arch;

//--------------------------------------------------------
class ArchEng {
public:
    ArchEng(const Arch& arch)
        : m_Arch(arch)
    {}

protected:
    const Arch& m_Arch;
};

}}

#endif

