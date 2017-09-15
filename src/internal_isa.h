#ifndef _INTERNAL_ISA_H
#define _INTERNAL_ISA_H

#include "isa.h"

#define INTERNAL_OPCODE_READ_IFMAP 0xff
typedef struct RdIfmapArgs {
   uint64_t    opcode                : OPCODE_BITS;
   uint64_t    address               : 32;
   char        fname[128];
} RdIfmapArgs;

#define INTERNAL_OPCODE_READ_FILTER 0xfe
typedef struct RdFilterArgs {
   uint64_t    opcode                : OPCODE_BITS;
   uint64_t    address               : 32;
   char        fname[128];
} RdFilterArgs;

#define INTERNAL_OPCODE_WRITE_OFMAP 0xfd
typedef struct WrOfmapArgs {
   uint64_t    opcode                : OPCODE_BITS;
   char        fname[128];
   uint64_t    address               : 32;
   uint64_t    i_n                   : 8;
   uint64_t    w_c                   : 8;
   uint64_t    w_m                   : 8;
   uint64_t    o_rows                : 8;
   uint64_t    o_cols                : 8;
   uint64_t    word_size             : 8;
} WrOfmapArgs;

#endif

