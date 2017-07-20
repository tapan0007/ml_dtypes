#include "pe_array.h"
#include "state_buffer.h"
#include <iostream>


int main()
{
    /* setup - later put this in a class? */
    ProcessingElementArray pe_array;
    StateBufferArray       state_buffer;

    for (int i = 0; i < state_buffer.num(); i++) {
        pe_array.connect_west(i, &state_buffer[i]);
    }


    for (int i = 0; i < 10; i++) {
        std::cout << "time = " << i << std::endl;
        pe_array.step();
        pe_array.dump(stdout);
    }
}

