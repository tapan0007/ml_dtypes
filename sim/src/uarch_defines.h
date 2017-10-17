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
        static const size_t row_partition_nbytes = 1 << (BANKS_PER_PARTITION_BITS + BANK_BITS);
        static const size_t column_partition_nbytes = 1 << (BANKS_PER_PARTITION_BITS + BANK_BITS);

};

#define UNUSED(X) (void)(X)

#endif
