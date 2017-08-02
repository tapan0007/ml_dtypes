#include "pe_array.h"
#include "state_buffer.h"
#include "sequencer.h"
#include "psum_buffer.h"
#include "activate.h"
#include "io.h"
#include <iostream>

#define STEP() \
    std::cout << "time = " << i << std::endl; \
    state_array.step_write(); \
    activate_array.step(); \
    psum_array.step(); \
    pe_array.step(); \
    state_array.step_read(); \
    sequencer.step(); \
    pe_array.dump(stdout); 


Memory memory = Memory(16*1024);

int main()
{
    /* setup - later put this in a class? */
    ProcessingElementArray pe_array;
    StateBufferArray       state_array;
    Sequencer              sequencer;
    PSumBufferArray        psum_array;
    ActivateArray          activate_array;
    ConvolveArgs cargs;
    int n_bytes;
    //int i = 0;
    int num_sb = state_array.num();


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

    /* load weights/filters */
    cargs.ifmap_addr = 0;
    n_bytes = memory.io_mmap(cargs.ifmap_addr, "/home/ec2-user/InklingUT/src/i_uint8_1x3x2x2.npy", 
                cargs.i_r, cargs.i_s, cargs.i_t, cargs.i_u);
    cargs.filter_addr = (cargs.ifmap_addr + n_bytes + 0x3ff) & ~0x3ff;
    n_bytes = memory.io_mmap(cargs.filter_addr, "/home/ec2-user/InklingUT/src/f_uint8_3x2x1x1.npy", 
            cargs.w_r, cargs.w_s, cargs.w_t, cargs.w_u);
    cargs.ofmap_addr = (cargs.filter_addr + n_bytes + 0x3ff) & ~0x3ff;

    memory.swap_axes(cargs.filter_addr, cargs.w_r, cargs.w_s, cargs.w_t, cargs.w_u, n_bytes); 
    cargs.weight_dtype = UINT8;


    /* set sequencer state */
    sequencer.convolve(cargs);
    for (int i = 0; i < 138; i++) {
        STEP();
    }
}

