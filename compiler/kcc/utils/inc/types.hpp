#ifndef KCC_UTILS_TYPES_H
#define KCC_UTILS_TYPES_H

namespace kcc {
namespace utils {

typedef signed char int8;
typedef short int16;
typedef int int32;
typedef long int64;
 
static_assert(sizeof(int8)  == 1, "sizeof(int8) != 1");
static_assert(sizeof(int16) == 2, "sizeof(int16) != 2");
static_assert(sizeof(int32) == 4, "sizeof(int32) != 4");
static_assert(sizeof(int64) == 8, "sizeof(int64) != 8");




typedef int32 LayerId;
const LayerId LayerId_Null = -1;

#if 0
typedef int64 StateBufferAddress;
const StateBufferAddress StateBufferAddress_Invalid = -1;
#endif

typedef int64 StateBufferAddress;
enum : StateBufferAddress {
    StateBufferAddress_Invalid = -1L
};

}}

#endif

