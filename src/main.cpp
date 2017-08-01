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


int main()
{
    /* setup - later put this in a class? */
    ProcessingElementArray pe_array;
    StateBufferArray       state_array;
    Sequencer              sequencer;
    PSumBufferArray        psum_array;
    ActivateArray          activate_array;
    int i = 0;
    int i_r,i_s,i_t,i_u;
    int w_r,w_s,w_t,w_u;
    addr_t ifmap_addr;
    addr_t filter_addr;
    int num_sb = state_array.num();


    /* make necessary connections */
    for (int j=0; j < pe_array.num_cols(); j++) {
        activate_array.connect_psum(j, &psum_array[j]);
    }
    psum_array.connect_west(&state_array[num_sb - 1]);
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
    ifmap_addr = (addr_t)io_mmap("/home/ec2-user/InklingUT/src/i_uint8_1x3x2x2.npy", i_r,i_s,i_t,i_u);
    filter_addr = (addr_t)io_mmap("/home/ec2-user/InklingUT/src/f_uint8_3x2x1x1.npy", w_r,w_s,w_t,w_u);


    /* set sequencer state */
    ARBPRECTYPE weight_dtype = UINT8;
    size_t filter_stride = sizeofArbPrecType(weight_dtype) * w_s * w_t * w_u;
    sequencer.set_clamp(false);
    sequencer.set_ifmap_valid(false);
    sequencer.set_weight_valid(true);
    sequencer.set_weight_addr(filter_addr + filter_stride - 1);
    sequencer.set_weight_stride(filter_stride);
    sequencer.set_weight_dtype(weight_dtype);
    sequencer.set_toggle_weight(false);
    sequencer.set_row_countdown(i_s);

    /* step in weights, clamp on last step */
    i = 0;
    for (; i < 2; i++) {
        if (i == 1) {
            sequencer.set_clamp(true);
        }
        STEP();
    }

    /* unclamp, stop feeding weights, feed ifmaps instead */
    ARBPRECTYPE psum_dtype = UINT32;
    sequencer.set_clamp(false);
    sequencer.set_ifmap_valid(true);
    sequencer.set_ifmap_addr(ifmap_addr);
    sequencer.set_ifmap_stride(sizeof(uint8_t) * i_t * i_u);
    sequencer.set_row_countdown(i_s);
    sequencer.set_start_psum(true);
    sequencer.set_end_psum(true);
    sequencer.set_psum_addr(0);
    sequencer.set_psum_stride(sizeofArbPrecType(psum_dtype) * w_s * i_t * i_u);
    sequencer.set_psum_dtype(UINT32);
    sequencer.set_column_countdown(2);
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
    sequencer.set_end_psum(false);
    sequencer.set_ifmap_valid(false);
    sequencer.set_weight_valid(false);
    for (; i < 128+8; i++) {
        STEP();
    }

}

