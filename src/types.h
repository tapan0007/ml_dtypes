#ifndef _TYPES_H
#define _TYPES_H

#include <stdint.h>
#include <stdlib.h>

enum ARBPRECTYPE {UINT8=0, UINT32, FP32, NUM_ARBPRECTYPE, INVALID_ARBPRECTYPE};

enum ADDRCLASS    {INVALID_ADDRCLASS=-1, WEIGHT_ADDR=0, IFMAP_ADDR, PSUM_ADDR, NUM_ADDRCLASS};

enum ACTIVATIONFUNCTION {INVALID_ACTIVATIONFUNCTION=-1, IDENTITY, RELU, LEAKY_RELU, SIGMIOD, TANH,  NUM_ACTIVATIONFUNCTION};

enum POOLTYPE {INVALID_POOLTYPE=-1, NO_POOL, AVG_POOL, MAX_POOL,  NUM_POOLTYPE};

typedef uint64_t addr_t;
#define MAX_ADDR UINTMAX_MAX


typedef uint64_t tick_t;
#define MAX_TICK UINTMAX_MAX

class Constants
{
    public:
        static const unsigned int rows;
        static const unsigned int columns;
        static const unsigned int banks_per_partition;
        static const size_t bytes_per_bank;
        static const size_t partition_nbytes;
};


#define UNUSED(X) (void)(X)

#endif
