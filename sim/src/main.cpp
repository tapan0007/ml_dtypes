#include "pe_array.h"
#include "state_buffer.h"
#include "sequencer.h"
#include "psum_buffer.h"
#include "pool.h"
#include "activate.h"
#include "io.h"
#include "string.h"
#include "ibufferfile.h"
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


/* globals! */
Memory memory = Memory(SZ(ROW_BITS) * SZ(ROW_SIZE_BITS) + SZ(COLUMN_BITS) * SZ(COLUMN_SIZE_BITS));
addr_t state_buffer_base = 0x0;
addr_t psum_buffer_base = SZ(ROW_BITS) * SZ(ROW_SIZE_BITS);


int main(int argc, char **argv)
{
    /* setup - later put this in a class? */
    ProcessingElementArray pe_array;
    StateBufferArray       state_array;
    Sequencer              sequencer = Sequencer();
    PSumBufferArray        psum_array;
    PoolArray              pool_array;
    ActivateArray          activate_array;
    int num_sb = state_array.num();
    UopFeedInterface *feed;


    /* Parse args */
    if (argc < 2) {
        printf("Usage is %s [object file]", argv[0]);
        return 0;
    }
    feed = new IBufferFile(argv[1]);


    /* make necessary connections */
    sequencer.connect_uopfeed(feed);
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


    //sequencer.convolve_dynamic(cargs, o_rows, o_cols);
    int i = 0;
    while (!sequencer.done()) {
        STEP();
        i++;
    }
    for (int j = 0; j < 128+64; j++) {
        STEP();
        i++;
    }
    free(feed);

}

