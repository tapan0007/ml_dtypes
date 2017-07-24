#ifndef _TYPES_H
#define _TYPES_H

enum ArbPrecType {UINT8=0, UINT32, FP32, NUM_ARBPRECTYPE};

typedef uint64_t addr_t;
#define MAX_ADDR UINTMAX_MAX

typedef uint64_t tick_t;
#define MAX_TICK UINTMAX_MAX

#endif
