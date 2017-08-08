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
    sequencer.step(); 
 //   pe_array.dump(stdout); 


Memory memory = Memory(16*1024);

int main(int argc, char **argv)
{
    /* setup - later put this in a class? */
    ProcessingElementArray pe_array;
    StateBufferArray       state_array;
    Sequencer              sequencer;
    PSumBufferArray        psum_array;
    ActivateArray          activate_array;
    ConvolveArgs cargs;
    int n_bytes;
    int r,s,t,u;
    //int i = 0;
    int num_sb = state_array.num();
    std::string i_name, f_name, o_name;
    if (argc < 4) {
        i_name = "/home/ec2-user/InklingUT/src/i_uint8_1x3x2x2.npy"; 
        f_name = "/home/ec2-user/InklingUT/src/f_uint8_2x3x1x1.npy"; 
        o_name = "/home/ec2-user/InklingUT/src/ofmap.npy"; 
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

    /* load weights/filters */
    cargs.ifmap_addr = 0;
    cargs.weight_dtype = UINT8;

    /* load io_mmap */
    n_bytes = memory.io_mmap(cargs.ifmap_addr, i_name, r, s, t, u);
    cargs.i_n = r;
    cargs.i_c = s;
    cargs.i_h = t;
    cargs.i_w = u;

    /* load filter */
    cargs.filter_addr = (cargs.ifmap_addr + n_bytes + 0x3ff) & ~0x3ff;
    n_bytes = memory.io_mmap(cargs.filter_addr, f_name, r, s, t, u);
    cargs.w_c = s; // for swap, M now corresponds to C
    cargs.w_m = r; // for swap, C now corresponds to M
    cargs.w_r = t;
    cargs.w_s = u;


    cargs.ofmap_addr = (cargs.filter_addr + n_bytes + 0x3ff) & ~0x3ff;
    memory.swap_axes(cargs.filter_addr, r, s, t, u, n_bytes);

    /* set sequencer state */
    sequencer.convolve(cargs);
    int i = 0;
    while (sequencer.steps_to_do()) {
        STEP();
        i++;
    }

    memory.io_write(o_name, cargs.ofmap_addr, cargs.i_n, cargs.w_m, (cargs.i_h - cargs.w_r + 1), (cargs.i_w - cargs.w_s + 1), UINT32);
}

