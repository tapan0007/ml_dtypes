#include "pe_array.h"
#include <iostream>


int main()
{
    ProcessingElementArray pe_array;
    for (int i = 0; i < 10; i++) {
        std::cout << "time = " << i;
        pe_array.step();
        pe_array.dump(stdout);
    }
}

