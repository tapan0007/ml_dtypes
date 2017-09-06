#ifndef ISA_H
#define ISA_H


typedef struct LdWeightsArgs {
   uint64_t    address               : 32;
   uint64_t    dtype                 : 8;
   uint64_t    x_step                : 12;
   uint64_t    x_num                 : 9;
   uint64_t    y_step                : 12;
   uint64_t    y_num                 : 9;
   uint64_t    num_rows              : 7;
} LdWeightsArgs;
#endif
