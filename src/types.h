#ifndef _TYPES_H
#define _TYPES_H

#include <stdint.h>
#include <stdlib.h>

enum ARBPRECTYPE {UINT8=0, UINT32, FP32, NUM_ARBPRECTYPE, INVALID_ARBPRECTYPE};

enum FMAPDTYPE {FINT4=0, FUINT4, FINT8, FUINT8, FINT16, FUINT16,FUINT32, FFP32, FFP16=7, NUM_FMAPDTYPE, INVALID_FMAPDTYPE};
enum WEIGHTDTYPE {WINT4=0, WUINT4, WINT8, WUINT8, WINT16, WUINT16, WUINT32, WFP32, WFP16=7, NUM_WEIGHTDTYPE, INVALID_WEIGHTDTYPE};

// FIXME -= make better way to do this
extern uint8_t fmapdtype_to_bytes[NUM_FMAPDTYPE];
extern uint8_t weightdtype_to_bytes[NUM_WEIGHTDTYPE];

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

typedef union {
    struct {
        uint8_t nibble_0 : 4;
        uint8_t nibble_1 : 4;
    };
    uint8_t byte;
} psum_data;


class Constants
{
    // TODO: find compile time log 2
    public:
        static const uint8_t row_bits = 6;
        static const uint8_t type_bits = 3;
        static const uint8_t rows = 1 << Constants::row_bits;
        static const uint8_t column_bits = 7;
        static const uint8_t columns = 1 << Constants::column_bits;
        static const uint8_t banks_per_partition_bits = 2;
        static const uint8_t banks_per_partition = (1 << Constants::banks_per_partition_bits);
        static const size_t bank_bits = 13;
        static const size_t bytes_per_bank = (1 << Constants::bank_bits);
        static const size_t addr_size = 20;
        static const size_t partition_nbytes = Constants::banks_per_partition * Constants::bytes_per_bank;
        static const size_t psum_buffer_width_bits = 6;
        static const size_t psum_buffer_width = (1 << Constants::psum_buffer_width_bits);
        static const size_t psum_buffer_entries_bits = 8;
        static const size_t psum_buffer_entries = (1 << Constants::psum_buffer_entries_bits);
        static const size_t psum_bank_bits = Constants::banks_per_partition;
        static const size_t psum_banks = (1 << Constants::psum_bank_bits);
        static const size_t psum_addr_bits = Constants::psum_buffer_width_bits + 
            Constants::psum_buffer_entries_bits + Constants::psum_bank_bits;
        static const size_t psum_addr = (1 << Constants::psum_addr_bits);

};


#define UNUSED(X) (void)(X)

#endif
