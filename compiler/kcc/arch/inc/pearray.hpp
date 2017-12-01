#pragma once

#ifndef KCC_ARCH_PEARRAY_H
#define KCC_ARCH_PEARRAY_H 1

#include <assert.h>

namespace kcc {
namespace arch {

//--------------------------------------------------------
class PeArray {
public:
    //----------------------------------------------------------------
    PeArray(int numberRows, int numberColumns);

    //----------------------------------------------------------------
    int gNumberRows() const {
        return m_NumberRows;
    }

    //----------------------------------------------------------------
    int gNumberColumns() const {
        return m_NumberColumns;
    }

    //----------------------------------------------------------------
    long gInstructionRamStartInBytes() const {
        return 0x001D00000L;
    }

    //----------------------------------------------------------------
    long gInstructionRamEndInBytes() const {
        return 0x001D03FFFL;
    }

private:
    int m_NumberRows;
    int m_NumberColumns;
};

}}

#endif // KCC_ARCH_PEARRAY_H

