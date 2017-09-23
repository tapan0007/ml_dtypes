#ifndef _TCC_H
#define _TCC_H

#include "types.h"
#include "stdio.h"


/* we don't have a DMA engine, so compile a special sim-only instruction to read
 * a numpy file (i_name) into the state buffer at ifmap_full_addr */
void compile_read_ifmap(FILE *file,
        addr_t ifmap_full_addr, const char *i_name);

void compile_read_filter(FILE *file,
        addr_t filter_full_addr, const char *i_name);

void compile_write_ofmap(FILE *file,
        const char *o_name, addr_t addr, uint64_t i_n, 
        uint64_t w_c,  uint64_t w_m,
        uint64_t o_rows, uint64_t o_cols, 
        size_t word_size);

/* convolve kernel */
void
compile_convolve(FILE *file,
        uint64_t &o_rows, uint64_t &o_cols,
        uint64_t ofmap_full_addr,
        uint64_t ifmap_full_addr, uint64_t idim[4],
        uint64_t filter_full_addr, uint64_t wdim[4],
        ARBPRECTYPE dtype,
        uint8_t padding[2], uint8_t stride[2], uint8_t dilate[2]);


#endif
