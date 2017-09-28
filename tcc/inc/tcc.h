#ifndef _TCC_H
#define _TCC_H

#include "dtypes.h"
#include "stdio.h"


/* we don't have a DMA engine, so compile a special sim-only instruction to read
 * a numpy file (i_name) into the state buffer at ifmap_full_addr */

void compile_read_ifmap(FILE *file,
        const addr_t ifmap_full_addr, const char *i_name);

void compile_read_filter(FILE *file,
        const addr_t filter_full_addr, const char *i_name);

void compile_write_ofmap(FILE *file,
        const char *o_name, const addr_t addr, const uint64_t i_n, 
        const uint64_t w_c,  const uint64_t w_m,
        const uint64_t o_rows, const uint64_t o_cols, 
        const size_t word_size);

/* convolve kernel */
void
compile_convolve(FILE *file,
        const uint64_t ifmap_addr, const uint64_t ifmap_dims[4],
        const uint64_t filter_addr, const uint64_t filter_dims[4],
        const uint64_t ofmap_addr, uint64_t ofmap_dims[4], /* output */
        const ARBPRECTYPE dtype,
        const uint8_t padding[2],  /* HW */
        const uint8_t stride[2],   /* HW */
        const uint8_t dilate[2]);  /* HW */

void
compile_pool(FILE *file,
        const uint64_t ifmap_addr, const uint64_t ifmap_dims[4],
        const uint64_t kernel_dims[4],
        const uint64_t ofmap_addr, uint64_t ofmap_dims[4], /* output */
        const uint64_t stride_dims[4],
        POOLFUNC pool_func);

#endif
