#include "pe_array.h"
#include "state_buffer.h"
#include "sequencer.h"
#include "psum_buffer.h"
#include "pool.h"
#include "activate.h"
#include "io.h"
#include "string.h"
#include "kernel_tester.h"
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
Memory memory = Memory(Constants::columns * Constants::partition_nbytes + Constants::columns * Constants::psum_addr);
addr_t state_buffer_base = 0x0;
addr_t psum_buffer_base = Constants::columns * Constants::partition_nbytes;

/* parse options */
UopFeedInterface *
parse_args(int argc, char **argv) {
    char *i_name, *f_name, *o_name;
    UopFeedInterface *feed = NULL;
    KernelTester *kernel_tester;
    uint8_t padding[2] = {0};
    int i = 1;

    if (argc < 2) {
        printf("Usage is %s [--kernel_test OR object_file]\n", argv[0]);
        return feed;
    }

    if (!strcmp(argv[i],"--kernel_test")) {
        if (argc < 4) {
            printf("Usage is %s --kernel_test [-p PAD] IFMAP_FILE FILTER_FILE OYUTPUT_FILE\n", argv[0]);
            return feed;
        }
        if (!strcmp(argv[++i], "-p")) {
            padding[1] = atoi(argv[++i]);
            padding[0] = atoi(argv[++i]);
            i++;
        }
        i_name = argv[i++];
        f_name = argv[i++];
        o_name = argv[i++];
        kernel_tester = new KernelTester();
        kernel_tester->convolve(i_name, f_name, o_name, padding);
        feed = kernel_tester;
    } else {
        if (argc < 2) {
            printf("Usage is %s [object file]", argv[0]);
            return feed;
        }
        i_name = argv[i];
        assert(0 && "not implemented");
    }
    return feed;
}


int main(int argc, char **argv)
{
    /* setup - later put this in a class? */
    ProcessingElementArray pe_array;
    StateBufferArray       state_array;
    Sequencer              sequencer = Sequencer();
    PSumBufferArray        psum_array;
    PoolArray              pool_array;
    ActivateArray          activate_array;
    UopFeedInterface *feed = NULL;
    int num_sb = state_array.num();

    if (!(feed = parse_args(argc, argv))) {
        return 1;
    }

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

