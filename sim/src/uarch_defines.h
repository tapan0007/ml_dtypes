#ifndef _UARCH_DEFINES_H
#define _UARCH_DEFINES_H

#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <uarch_cfg.h>


typedef uint64_t tick_t;
#define MAX_TICK UINTMAX_MAX


class Constants
{
    // TODO: find compile time log 2
    public:
        static const uint8_t type_bits = 8;
        static const uint8_t rows = 1 << ROW_BITS;
        static const uint8_t columns = 1 << COLUMN_BITS;
        static const uint8_t banks_per_partition = 1 << 
            BANKS_PER_PARTITION_BITS;
        static const size_t bytes_per_bank = 1 << BANK_BITS;
        static const size_t partition_nbytes = Constants::banks_per_partition 
            * Constants::bytes_per_bank;
        static const size_t psum_buffer_width_bits = 6; // 64 byte entries
        static const size_t psum_buffer_width = 1 << 
            Constants::psum_buffer_width_bits;
        static const size_t psum_buffer_entries_bits = 8; // 256 entries
        static const size_t psum_buffer_entries = 1 << 
            Constants::psum_buffer_entries_bits;
        static const size_t psum_banks = 1 << BANKS_PER_PARTITION_BITS;
        static const size_t psum_addr_bits = Constants::psum_buffer_width_bits 
            + Constants::psum_buffer_entries_bits +BANKS_PER_PARTITION_BITS;
        static const size_t psum_addr = 1 << Constants::psum_addr_bits;

};

#define UNUSED(X) (void)(X)

#endif
