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
        const ARBPRECTYPE dtype);

/*[ifmap/filter]_addrs are arrays of statebuffer addresses.  Arrays 
 * deal with cases iwhen with the number of ifmap channels is > number of rows.
 * In this case, the ifmaps and filters must be "wrapped".  Each address in the
 * array is the wrap offset */
void
compile_convolve(FILE *out_binary,
        const addr_t *ifmap_addrs, const uint64_t ifmap_dims[4], /* NCHW */
        const addr_t *filter_addr, const uint64_t filter_dims[4], /* MCRS */
        const addr_t ofmap_addr, uint64_t ofmap_dims[4], /* output NCHW */
        const ARBPRECTYPE in_dtype, const ARBPRECTYPE out_dtype,
        const uint8_t padding[2],  /* Height,Width */
        const uint8_t stride[2],   /* Height,Width */
        const uint8_t dilate[2]);  /* Height,Width */

void
compile_pool(FILE *out_binary,
        const addr_t ifmap_addr, const uint64_t ifmap_dims[4], /* NCHW */
        const uint64_t kernel_dims[4], /* NCHW */
        const addr_t ofmap_addr, uint64_t ofmap_dims[4], /* output NCHW */
        const uint64_t stride_dims[4], /* NCHW */
        const ARBPRECTYPE dtype,
        POOLFUNC pool_func);
 
void
compile_activation(FILE *out_binary,
        const addr_t ifmap_addr, const uint64_t ifmap_dims[4], /* NCHW */
        const addr_t ofmap_addr, uint64_t ofmap_dims[4], /* output NCHW */
        const ARBPRECTYPE in_dtype,
        const ARBPRECTYPE out_dtype,
        ACTIVATIONFUNC act_func);
#if 0
void
compile_resadd(FILE *out_binary,
        const addr_t lhs_addr, 
        const addr_t rhs_addr,
        const uint64_t dims[4],
        const ARBPRECTYPE dtype);
#endif 

#endif
