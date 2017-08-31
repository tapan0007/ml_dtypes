#include "pe_array.h"
#include "state_buffer.h"
#include "sequencer.h"
#include "psum_buffer.h"
#include "pool.h"
#include "activate.h"
#include "io.h"
#include <iostream>

#define STEP() \
    std::cout << "time = " << i << std::endl; \
    state_array.step_write(); \
    activate_array.step(); \
    pool_array.step(); \
    psum_array.step(); \
    pe_array.step(); \
    state_array.step_read(); \
    sequencer.step(); 
 //   pe_array.dump(stdout); 


Memory memory = Memory(Constants::columns * Constants::partition_nbytes + Constants::columns * Constants::psum_addr);
addr_t state_buffer_base = 0x0;
addr_t psum_buffer_base = Constants::columns * Constants::partition_nbytes;

int main(int argc, char **argv)
{
    /* setup - later put this in a class? */
    ProcessingElementArray pe_array;
    StateBufferArray       state_array;
    Sequencer              sequencer;
    PSumBufferArray        psum_array;
    PoolArray              pool_array;
    ActivateArray          activate_array;
    ConvolveArgs cargs;
    void *f_ptr, *i_ptr, *o_ptr;
    int r,s,t,u;
    size_t word_size;
    //int i = 0;
    int num_sb = state_array.num();
    std::string i_name, f_name, o_name;
    if (argc < 4) {
        i_name = "/home/ec2-user/InklingTest/input/ifmaps/i_uint8_1x3x2x2_rand.npy"; 
        f_name = "/home/ec2-user/InklingTest/input/filters/f_uint8_2x3x1x1_rand.npy"; 
        o_name = "/home/ec2-user/InklingUT/ofmap.npy"; 
    } else {
        i_name = argv[1];
        f_name = argv[2];
        o_name = argv[3];
    }


    /* make necessary connections */
    for (int j=0; j < pe_array.num_cols(); j++) {
        activate_array.connect_psum(j, &psum_array[j]);
    }
    psum_array.connect_west(state_array.get_edge());
    int last_row = pe_array.num_rows()-1;
    for (int j=0; j < pe_array.num_cols(); j++) {
        psum_array.connect_north(j, &pe_array[last_row][j]);
    }
    for (int j=0; j < num_sb; j++) {
        pe_array.connect_west(j, &state_array[j]);
        pe_array.connect_statebuffer(j, &state_array[j]);
        state_array.connect_activate(j, &activate_array[j]);
    }
    state_array.connect_north(&sequencer);
    pool_array.connect(&sequencer);

    /* load weights/filters */
    cargs.ifmap_full_addr = 0;
    cargs.filter_full_addr = 2 * Constants::bytes_per_bank;
    cargs.weight_dtype = UINT8;

    /* load io_mmap */
    i_ptr = memory.io_mmap(i_name, r, s, t, u, word_size);
    cargs.i_n = r;
    cargs.i_c = s;
    cargs.i_h = t;
    cargs.i_w = u;
    assert(cargs.i_n == 1 && "cannot support multibatching yet");
    memory.bank_mmap(cargs.ifmap_full_addr, i_ptr, cargs.i_c, cargs.i_h * cargs.i_w * word_size);

    /* load filter */
    cargs.filter_full_addr = 1 * Constants::bytes_per_bank;
    f_ptr = memory.io_mmap(f_name, r, s, t, u, word_size);
    memory.swap_axes(f_ptr, r, s, t, u, word_size);
    cargs.w_c = s; // for swap, M now corresponds to C
    cargs.w_m = r; // for swap, C now corresponds to M
    cargs.w_r = t;
    cargs.w_s = u;
    memory.bank_mmap(cargs.filter_full_addr, f_ptr, cargs.w_c, cargs.w_m * cargs.w_r * cargs.w_s * word_size);


    /* set sequencer state */
    //sequencer.convolve_static(cargs);
    sequencer.convolve_dynamic(cargs);
    int i = 0;
    while (!sequencer.done()) {
        STEP();
        i++;
    }
    for (int j = 0; j < 128+64; j++) {
        STEP();
        i++;
    }

    int o_rows = (cargs.i_h - cargs.w_r + 1);
    int o_cols = (cargs.i_w - cargs.w_s + 1);
    word_size = 4; // HACKE DIN FIX, outputting 32 
    o_ptr = memory.psum_bank_munmap(psum_buffer_base, cargs.w_c, o_rows * o_cols * word_size);
    memory.io_write(o_name, o_ptr, cargs.i_n, cargs.w_m, o_rows, o_cols, word_size);
}

