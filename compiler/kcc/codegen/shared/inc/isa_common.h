#ifndef ISA_COMMON_H
#define ISA_COMMON_H

#include <assert.h> 

#define TONGA_PACKED __attribute__((__packed__))
#define tonga_assert(x)    assert(x)

#define BIT(n)                  ( 1<<(n) )
#define BIT_MASK(len)           ( BIT(len)-1 )
#define BITFIELD_MASK(start, len)     ( BIT_MASK(len)<<(start) )
#define BITFIELD_EXTRACT(x, start, len)   ( ((x)>>(start)) & BIT_MASK(len) )


#endif
