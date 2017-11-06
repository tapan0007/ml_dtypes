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
    pool_array.step(); \
    psum_array.step(); \
    pe_array.step(); \
    state_array.step_read(); \
    sequencer.step(); 
 //   pe_array.dump(stdout); 
//    activate_array.step(); 




int main(int argc, char **argv)
{
    /* setup - later put this in a class? */
    size_t num_rows = SZ(ROW_BITS);
    size_t num_cols = SZ(COLUMN_BITS);
    Memory memory = Memory(0x04000000);
    MemoryMap mmap = MemoryMap(&memory);
    ProcessingElementArray pe_array;
    StateBufferArray       state_array = 
        StateBufferArray(&mmap, MMAP_SB_BASE, num_rows);
    Sequencer              sequencer = Sequencer(&memory);
    PSumBufferArray        psum_array = 
        PSumBufferArray(&mmap, MMAP_PSUM_BASE, num_cols);;
    PoolArray              pool_array = PoolArray(&mmap, num_cols);
    ActivateArray          activate_array;
    UopFeedInterface *feed;


    /* Parse args */
    if (argc < 2) {
        printf("Usage is %s [object file]", argv[0]);
        return 0;
    }
    feed = new IBufferFile(argv[1]);


    /* make necessary connections */
    sequencer.connect_uopfeed(feed);
    psum_array.connect_west(state_array.get_edge());
    int last_row = pe_array.num_rows()-1;
    for (size_t j=0; j < num_cols; j++) {
        psum_array.connect_north(j, &pe_array[last_row][j]);
    }
    for (size_t j=0; j < num_rows; j++) {
        pe_array.connect_west(j, &state_array[j]);
        pe_array.connect_statebuffer(j, &state_array[j]);
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

}

