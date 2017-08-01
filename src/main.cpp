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
    int i = 0;
    int i_r,i_s,i_t,i_u;
    int w_r,w_s,w_t,w_u;
    addr_t ifmap_addr, filter_addr, ofmap_addr;
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
    ifmap_addr = 0;
    filter_addr = (memory.io_mmap(ifmap_addr, "/home/ec2-user/InklingUT/src/i_uint8_1x3x2x2.npy", i_r,i_s,i_t,i_u) + 0x3ff) & ~0x3ff;
    ofmap_addr = (filter_addr + memory.io_mmap(filter_addr, "/home/ec2-user/InklingUT/src/f_uint8_3x2x1x1.npy", w_r,w_s,w_t,w_u) + 0x3ff) & ~0x3ff;


    /* set sequencer state */
    ARBPRECTYPE weight_dtype = UINT8;
    size_t filter_stride = sizeofArbPrecType(weight_dtype) * w_s * w_t * w_u;
    sequencer.edge_signals.weight_clamp = false;
    sequencer.edge_signals.ifmap_valid = false;
    sequencer.edge_signals.weight_valid = true;
    sequencer.edge_signals.weight_addr = filter_addr + filter_stride - 1;
    sequencer.edge_signals.weight_stride = filter_stride;
    sequencer.edge_signals.weight_dtype = weight_dtype;
    sequencer.edge_signals.weight_toggle = false;
    sequencer.edge_signals.row_countdown = i_s;

    /* step in weights, weight_clamp on last step */
    i = 0;
    for (; i < 2; i++) {
        if (i == 1) {
            sequencer.edge_signals.weight_clamp = true;
        }
        STEP();
        sequencer.edge_signals.weight_addr -= sizeofArbPrecType(sequencer.edge_signals.weight_dtype);
    }

    /* unweight_clamp, stop feeding weights, feed ifmaps instead */
    ARBPRECTYPE psum_dtype = UINT32;
    sequencer.edge_signals.weight_clamp = false;
    sequencer.edge_signals.ifmap_valid = true;
    sequencer.edge_signals.ifmap_addr = ifmap_addr;
    sequencer.edge_signals.ifmap_stride = sizeofArbPrecType(UINT8) * i_t * i_u;
    sequencer.edge_signals.row_countdown = i_s;
    sequencer.edge_signals.psum_start = true;
    sequencer.edge_signals.psum_end = true;
    sequencer.edge_signals.psum_id = 0;
    sequencer.edge_signals.ofmap_addr = ofmap_addr;
    sequencer.edge_signals.ofmap_stride = sizeofArbPrecType(psum_dtype) * w_s * i_t * i_u;
    sequencer.edge_signals.psum_dtype = psum_dtype;
    sequencer.edge_signals.column_countdown = 2;
    sequencer.edge_signals.weight_valid = false;
    sequencer.edge_signals.weight_toggle = true;
    sequencer.edge_signals.activation_valid = true;
    sequencer.edge_signals.activation_valid = IDENTITY;
    sequencer.edge_signals.pool_valid = true;
    sequencer.edge_signals.pool_type = NO_POOL;
    sequencer.edge_signals.pool_dtype = psum_dtype;

    /* unweight_clamp, done toggling */
    for (; i < 6; i++) {
        STEP();
        if (i == 2) {
            sequencer.edge_signals.weight_clamp = false;
            sequencer.edge_signals.weight_toggle = false;
        }
        sequencer.edge_signals.psum_id++;
        sequencer.edge_signals.ifmap_addr += sizeofArbPrecType(UINT8);
        sequencer.edge_signals.ofmap_addr += sizeofArbPrecType(UINT32);
    }
    /* drain out */
    sequencer.edge_signals.psum_start = false;
    sequencer.edge_signals.psum_end = false;
    sequencer.edge_signals.ifmap_valid = false;
    sequencer.edge_signals.weight_valid = false;
    for (; i < 128+8; i++) {
        STEP();
    }

}

