#ifndef UARCH_CFG_H
#define UARCH_CFG_H

#define UNUSED(X) (void)(X)

#define SZ(BITS) (1 << (BITS))

#define ROW_BITS                    7
#define ROW_BYTE_OFFSET_BITS        3
#define BANKS_PER_ROW_BITS          2
#define ROWS_PER_BANK_PER_ROW_BITS  11
#define ROW_SIZE_BITS               ROW_BYTE_OFFSET_BITS + \
                                    BANKS_PER_ROW_BITS + \
                                    ROWS_PER_BANK_PER_ROW_BITS

#define COLUMN_BITS                 6
#define COLUMN_BYTE_OFFSET_BITS     13
#define BANKS_PER_COLUMN_BITS       2
#define COLUMN_META_BIT             1
#define COLUMN_SIZE_BITS            COLUMN_BYTE_OFFSET_BITS + \
                                    BANKS_PER_COLUMN_BITS + \
                                    COLUMN_META_BIT

#define PSUM_ENTRY_BITS             3

#define SB_SIZE_BITS                ROW_BITS + ROW_SIZE_BITS
#define PSUM_SIZE_BITS              COLUMN_BITS + COLUMN_SIZE_BITS

#define MMAP_SB_BASE                0
#define MMAP_PSUM_BASE              SZ(SB_SIZE_BITS)

#define ADDRESS_BITS                32
#define INSTRUCTION_NBYTES          256
#endif

