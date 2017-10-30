#ifndef UARCH_CFG_H
#define UARCH_CFG_H

#define UNUSED(X) (void)(X)

#define SZ(BITS) (1 << (BITS))

#define BANK_BITS 13

#define ROW_BITS                    7
#define BANKS_PER_ROW_BITS          2
#define SRAMS_PER_BANK_PER_ROW_BITS 2
#define ROW_SIZE_BITS               BANK_BITS + BANKS_PER_ROW_BITS + \
                                    SRAMS_PER_BANK_PER_ROW_BITS
#define SB_SIZE_BITS                ROW_BITS + ROW_SIZE_BITS

#define COLUMN_BITS                 6
#define BANKS_PER_COLUMN_BITS       4
#define SRAMS_PER_BANK_PER_COLUMN_BITS 2
#define COLUMN_SIZE_BITS            BANK_BITS + BANKS_PER_COLUMN_BITS
#define PSUM_SIZE_BITS              ROW_BITS + ROW_SIZE_BITS

#define MMAP_SB_BASE                0
#define MMAP_PSUM_BASE              SZ(SB_SIZE_BITS)

#define ADDRESS_BITS                32
#define INSTRUCTION_NBYTES          256
#endif

