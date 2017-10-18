#ifndef _TCC_H
#define _TCC_H

#include <stdio.h>
#include "tpb_isa.h"


void compile_read_ifmap(FILE *out_binary,
        const addr_t ifmap_sb_addr, const char *in_numpy_fname,
        const char *numpy_layout);

void compile_read_filter(FILE *out_binary,
        const addr_t filter_sb_addr, const char *in_numpy_fname, 
        const char *numpy_layout);

void compile_write_ofmap(FILE *out_binary,
        const char *out_numpy_name, const addr_t ofmap_sb_addr,
        const uint64_t dims[4],
        const size_t word_size);

void
compile_convolve(FILE *out_binary,
        const addr_t *ifmap_addrs, const uint64_t ifmap_dims[4],
        const addr_t *filter_addr, const uint64_t filter_dims[4],
        const addr_t ofmap_addr, uint64_t ofmap_dims[4], /* output */
        const ARBPRECTYPE dtype,
        const uint8_t padding[2],  /* Height,Width */
        const uint8_t stride[2],   /* Height,Width */
        const uint8_t dilate[2]);  /* Height,Width */

void
compile_pool(FILE *out_binary,
        const addr_t ifmap_addr, const uint64_t ifmap_dims[4],
        const uint64_t kernel_dims[4],
        const addr_t ofmap_addr, uint64_t ofmap_dims[4], /* output */
        const uint64_t stride_dims[4],
        const ARBPRECTYPE dtype,
        POOLFUNC pool_func);
 
void
compile_resadd(FILE *out_binary,
        const addr_t lhs_addr, 
        const addr_t rhs_addr,
        const uint64_t dims[4],
        const ARBPRECTYPE dtype);
 
 
#endif
