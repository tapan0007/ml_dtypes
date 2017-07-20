#ifndef STATE_BUFFER_H
#define STATE_BUFFER_H

#include "sigint.h"
#include <vector>

class StateBuffer : public EWInterface {
    public:
        StateBuffer();
        ~StateBuffer();
        EWSignals pull_ew();
    private:
        RandomInterfaceGenerator generator;
};

class StateBufferArray {
    public:
        StateBufferArray(int _num_buffers = 128);
        ~StateBufferArray();
        StateBuffer& operator[](int index);
        int num();
    private:
        std::vector<StateBuffer> buffers;
        int num_buffers;
};


#endif 
