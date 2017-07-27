#include "pe_array.h"
#include "state_buffer.h"
#include "sequencer.h"
#include "psum_buffer.h"
#include "io.h"
#include <iostream>

#define STEP() \
    std::cout << "time = " << i << std::endl; \
    psum_array.step(); \
    state_array.step(); \
    pe_array.step(); \
    sequencer.step(); \
    pe_array.dump(stdout); 


int main()
{
    /* setup - later put this in a class? */
    ProcessingElementArray pe_array;
    StateBufferArray       state_array;
    Sequencer              sequencer;
    PSumBufferArray        psum_array;
    int i = 0;
    int r,s,t,u;
    uint8_t *ifmap_addr;
    uint8_t *filter_addr;
    int num_sb = state_array.num();


    /* make necessary connections */
    psum_array.connect_west(&state_array[num_sb - 1]);
    int last_row = pe_array.num_rows()-1;
    for (int j=0; j < pe_array.num_cols(); j++) {
        psum_array.connect_north(j, &pe_array[last_row][j]);
    }
    for (int j=0; j < num_sb; j++) {
        pe_array.connect_west(j, &state_array[j]);
        pe_array.connect_statebuffer(j, &state_array[j]);
    }
    state_array.connect_north(&sequencer);

    /* load weights/filters */
    ifmap_addr = (uint8_t *)io_mmap("/home/ec2-user/InklingUT/src/i_uint8_1x3x2x2.npy", r,s,t,u);
    state_array.load_ifmap(ifmap_addr, 0, 3, sizeof(uint8_t) * t * u);
    filter_addr = (uint8_t *)io_mmap("/home/ec2-user/InklingUT/src/f_uint8_3x2x1x1.npy", r,s,t,u);
    state_array.load_weights(filter_addr, 0, 3, sizeof(uint8_t) * s * t * u, UINT8);


    /* set sequencer state */
    sequencer.set_clamp(false);
    sequencer.set_ifmap_valid(false);
    sequencer.set_weight_valid(true);
    sequencer.set_toggle_weight(false);

    /* step in weights, clamp on last step */
    i = 0;
    for (; i < 2; i++) {
        if (i == 1) {
            sequencer.set_clamp(true);
        }
        STEP();
    }

    /* unclamp, stop feeding weights, feed ifmaps instead */
    sequencer.set_clamp(false);
    sequencer.set_psum_addr(0);
    sequencer.set_ifmap_valid(true);
    sequencer.set_start_psum(true);
    sequencer.set_end_psum(2);
    sequencer.set_psum_dtype(UINT32);
    sequencer.set_weight_valid(false);
    sequencer.set_toggle_weight(true);
    STEP();
    i++;

    /* unclamp, done toggling */
    sequencer.set_clamp(false);
    sequencer.set_toggle_weight(false);
    for (; i < 6; i++) {
        STEP();
    }
    /* drain out */
    sequencer.set_start_psum(false);
    sequencer.set_end_psum(0);
    sequencer.set_ifmap_valid(false);
    sequencer.set_weight_valid(false);
    for (; i < 128+8; i++) {
        STEP();
    }

}

