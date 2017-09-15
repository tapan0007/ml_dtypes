#ifndef _COMPILER_H
#define _COMPILER_H

#include "types.h"

uint8_t
get_tile_type(uint8_t row, uint8_t col, 
        uint8_t n_rows, uint8_t n_cols);

void compile_read_ifmap(void **v_dest, size_t &dest_size, 
        addr_t ifmap_full_addr, char *i_name);

void compile_read_filter(void **v_dest, size_t &dest_size, 
        addr_t filter_full_addr, char *i_name);

void compile_write_ofmap(void **v_dest, size_t &dest_size, 
        char *o_name, addr_t addr, uint64_t i_n, 
        uint64_t w_c,  uint64_t w_m,
        uint64_t o_rows, uint64_t o_cols, 
        size_t word_size);
void
compile_convolve(void **v_dest, size_t &dest_size, 
        uint64_t &o_rows, uint64_t &o_cols,
        uint64_t ofmap_full_addr,
        uint64_t ifmap_full_addr, uint64_t idim[4],
        uint64_t filter_full_addr, uint64_t wdim[4],
        ARBPRECTYPE dtype,
        uint8_t padding[2], uint8_t stride[2], uint8_t dilate[2]);


#endif
