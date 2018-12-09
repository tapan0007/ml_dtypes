#pragma once

#ifndef KCC_UTILS_MISC_H
#define KCC_UTILS_MISC_H

#include "utils/inc/types.hpp"

namespace kcc {

template <typename T, int N>
constexpr size_t ArraySizeof(T (&)[N])
{
    return N;
}

static inline kcc_int64
roundUpToMultipleOfM(kcc_int64 k, kcc_int64 M)
{
    return ((k + M - 1)/M) * M;
}

} // namespace kcc

#endif
