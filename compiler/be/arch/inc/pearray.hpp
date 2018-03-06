#pragma once

#ifndef KCC_ARCH_PEARRAY_H
#define KCC_ARCH_PEARRAY_H 1


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

private:
    const kcc_int32 m_NumberRows;
    const kcc_int32 m_NumberColumns;
};

}}

#endif // KCC_ARCH_PEARRAY_H

