#pragma once

#ifndef KCC_UTILS_MISC_H
#define KCC_UTILS_MISC_H

#include "utils/inc/types.hpp"

namespace kcc {

static inline kcc_int64 
roundUpToMultipleOfM(kcc_int64 k, kcc_int64 M)
{
    return ((k + M - 1)/M) * M;
}

} // namespace kcc

#endif
