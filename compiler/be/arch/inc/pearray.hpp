#pragma once

#ifndef KCC_ARCH_PEARRAY_H
#define KCC_ARCH_PEARRAY_H 1

#include <assert.h>

#include "utils/inc/types.hpp"

namespace kcc {
namespace arch {

//--------------------------------------------------------
class PeArray {
public:
    //----------------------------------------------------------------
    PeArray(kcc_int32 numberRows, kcc_int32 numberColumns);

    //----------------------------------------------------------------
    kcc_int32 gNumberRows() const {
        return m_NumberRows;
    }

    //----------------------------------------------------------------
    kcc_int32 gNumberColumns() const {
        return m_NumberColumns;
    }

    //----------------------------------------------------------------
    kcc_int64 gInstructionRamStartInBytes() const {
        return 0x001D00000L;
    }

    //----------------------------------------------------------------
    kcc_int64 gInstructionRamEndInBytes() const {
        return 0x001D03FFFL;
    }

private:
    const kcc_int32 m_NumberRows;
    const kcc_int32 m_NumberColumns;
};

}}

#endif // KCC_ARCH_PEARRAY_H

