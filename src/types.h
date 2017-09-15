#ifndef _TYPES_H
#define _TYPES_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>

enum ARBPRECTYPE {INVALID_ARBPRECTYPE = 0, 
    INT8 = 2,   UINT8 = 3, 
    INT16 = 4,  UINT16 = 5, 
    FP16 = 7,
    INT32,      UINT32,
    FP32, 
    INT64 = 12, UINT64 = 13, 
    NUM_ARBPRECTYPE = 16};

ARBPRECTYPE get_upcast(ARBPRECTYPE);
size_t sizeofArbPrecType(ARBPRECTYPE type);


enum ACTIVATIONFUNCTION {INVALID_ACTIVATIONFUNCTION=-1, IDENTITY, RELU, LEAKY_RELU, SIGMIOD, TANH,  NUM_ACTIVATIONFUNCTION};


#define N_FLAG 1
#define S_FLAG 1 << 1
#define E_FLAG 1 << 2
#define W_FLAG 1 << 3

enum NSEW {N=0, S, E, W, NUM_NSEW};


enum POOLFUNC {
    MAX_POOL = 0, 
    AVG_POOL = 1, 
    IDENTITY_POOL = 2,
    NUM_POOLTYPE=3};

typedef uint64_t addr_t;
#define MAX_ADDR UINTMAX_MAX


typedef uint64_t tick_t;
#define MAX_TICK UINTMAX_MAX

#define ADDR_UNION(PREF)  \
    union { \
        struct { \
            uint64_t PREF##_addr : Constants::bank_bits; \
            uint64_t PREF##_bank : Constants::addr_size - Constants::bank_bits; \
        }; \
        uint64_t PREF##_full_addr : Constants::addr_size; \
    }; \

#define PSUM_UNION(PREF)  \
    union { \
        struct { \
            uint64_t PREF##_addr : Constants::psum_buffer_entries_bits + Constants::psum_buffer_width_bits; \
            uint64_t PREF##_bank : Constants::psum_bank_bits; \
        }; \
        uint64_t PREF##_full_addr : Constants::psum_bank_bits + Constants::psum_buffer_entries_bits + Constants::psum_buffer_width_bits; \
    }; \

#define BANKS_PER_PARTITION_BITS 2
#define INSTRUCTION_NBYTES 256
#define SB_SIZE (1 << COLUMN_BITS) * (1<<BANKS_PER_PARTITION_BITS) * ( 1 << BANK_BITS)
#define MMAP_PSUM_BASE SB_SIZE
class Constants
{
    // TODO: find compile time log 2
    public:
        static const uint8_t row_bits = 7;
        static const uint8_t type_bits = 8;
        static const uint8_t rows = 1 << Constants::row_bits;
        static const uint8_t column_bits = 6;
        static const uint8_t columns = 1 << Constants::column_bits;
        static const uint8_t banks_per_partition_bits = 2;
        static const uint8_t banks_per_partition = (1 << Constants::banks_per_partition_bits);
        static const size_t bank_bits = 13;
        static const size_t bytes_per_bank = (1 << Constants::bank_bits);
        static const size_t addr_size = 28;
        static const size_t partition_nbytes = Constants::banks_per_partition * Constants::bytes_per_bank;
        static const size_t psum_buffer_width_bits = 6;
        static const size_t psum_buffer_width = (1 << Constants::psum_buffer_width_bits);
        static const size_t psum_buffer_entries_bits = 8;
        static const size_t psum_buffer_entries = (1 << Constants::psum_buffer_entries_bits);
        static const size_t psum_bank_bits = Constants::banks_per_partition_bits;
        static const size_t psum_banks = (1 << Constants::psum_bank_bits);
        static const size_t psum_addr_bits = Constants::psum_buffer_width_bits + 
            Constants::psum_buffer_entries_bits + Constants::psum_bank_bits;
        static const size_t psum_addr = (1 << Constants::psum_addr_bits);
        static const size_t tile_size = floor(sqrt(Constants::psum_buffer_entries));

};

#define UNUSED(X) (void)(X)

#endif
