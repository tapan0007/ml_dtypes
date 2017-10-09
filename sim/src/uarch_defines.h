#ifndef _UARCH_DEFINES_H
#define _UARCH_DEFINES_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>


typedef uint64_t tick_t;
#define MAX_TICK UINTMAX_MAX


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
