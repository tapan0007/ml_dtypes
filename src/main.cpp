#include "pe_array.h"
#include "state_buffer.h"
#include "sequencer.h"
#include <iostream>


int main()
{
    /* setup - later put this in a class? */
    ProcessingElementArray pe_array;
    StateBufferArray       state_array;
    Sequencer              sequencer;

    /* make necessary connections */
    for (int i = 0; i < state_array.num(); i++) {
        pe_array.connect_west(i, &state_array[i]);
    }
    pe_array.connect_sequencer(&sequencer);
    state_array.connect_north(&sequencer);

    /* set sequencer state */
    sequencer.set_opcode(START_CALC);
    sequencer.set_psum_addr(0);
    sequencer.set_clamp_time(64);

    for (int i = 0; i < 100; i++) {
        std::cout << "time = " << i << std::endl;
        state_array.step();
        pe_array.step();
        sequencer.step();
        pe_array.dump(stdout);
    }
}

