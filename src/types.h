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

#define ADDR_UNION(PREF)  \
    union { \
        struct { \
            uint64_t PREF##_addr : Constants::bank_shift; \
            uint64_t PREF##_bank : Constants::addr_size - Constants::bank_shift; \
        }; \
        uint64_t PREF##_full_addr : Constants::addr_size; \
    }; \

class Constants
{
    public:
        static const unsigned int rows = 64;
        static const unsigned int columns = 128;
        static const unsigned int banks_per_partition = 4;
        static const size_t bank_shift = 13;
        static const size_t bytes_per_bank = (1 << Constants::bank_shift);
        static const size_t addr_size = 20;
        static const size_t partition_nbytes = Constants::banks_per_partition * Constants::bytes_per_bank;
};


#define UNUSED(X) (void)(X)

#endif
