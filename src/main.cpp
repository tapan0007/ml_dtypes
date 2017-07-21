#include "pe_array.h"
#include "state_buffer.h"
#include "sequencer.h"
#include <iostream>

#define STEP() \
    std::cout << "time = " << i << std::endl; \
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
    int i = 0;

    /* make necessary connections */
    for (int j=0; j < state_array.num(); j++) {
        pe_array.connect_west(j, &state_array[j]);
    }
    pe_array.connect_sequencer(&sequencer);
    state_array.connect_north(&sequencer);

    /* set sequencer state */
    sequencer.set_clamp(false);
    sequencer.set_ifmap_valid(false);
    sequencer.set_weight_valid(true);
    sequencer.set_toggle_weight(false);

    i = 0;
    for (; i < 64; i++) {
        STEP();
    }

    /* clamp, stop feeding weights, feed ifmaps instead */
    sequencer.set_clamp(true);
    sequencer.set_psum_addr(0);
    sequencer.set_ifmap_valid(true);
    sequencer.set_weight_valid(false);
    sequencer.set_toggle_weight(true);
    STEP();
    i++;

    /* unclamp, done toggling */
    sequencer.set_clamp(false);
    sequencer.set_toggle_weight(false);
    for (; i < 128; i++) {
        STEP();
    }
}

