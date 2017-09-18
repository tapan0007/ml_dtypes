#ifndef _TCC_H
#define _TCC_H

#include "types.h"

/* v_dest is the address to the pointer in memory where the instructions will be
 * written.  After the instructions are written, it is updated to point to the
 * next available memory location for writing instructions 
 *
 * dest_size is a reference that tells how many bytes were written.  I don't use
 * this for anything right now - we might delete it? 
 *
 * the remaining arguments are the arguments to the kernel.  
 */

/* we don't have a DMA engine, so compile a special sim-only instruction to read
 * a numpy file (i_name) into the state buffer at ifmap_full_addr */
void compile_read_ifmap(void **v_dest, size_t &dest_size, 
        addr_t ifmap_full_addr, const char *i_name);

void compile_read_filter(void **v_dest, size_t &dest_size, 
        addr_t filter_full_addr, const char *i_name);

void compile_write_ofmap(void **v_dest, size_t &dest_size, 
        const char *o_name, addr_t addr, uint64_t i_n, 
        uint64_t w_c,  uint64_t w_m,
        uint64_t o_rows, uint64_t o_cols, 
        size_t word_size);

/* convolve kernel */
void
compile_convolve(void **v_dest, size_t &dest_size, 
        uint64_t &o_rows, uint64_t &o_cols,
        uint64_t ofmap_full_addr,
        uint64_t ifmap_full_addr, uint64_t idim[4],
        uint64_t filter_full_addr, uint64_t wdim[4],
        ARBPRECTYPE dtype,
        uint8_t padding[2], uint8_t stride[2], uint8_t dilate[2]);


#endif
