#ifndef STATE_BUFFER_H
#define STATE_BUFFER_H

#include "sigint.h"
#include <vector>

class StateBuffer : public EWInterface, public StateBufferShiftInterface {
    public:
        StateBuffer();
        ~StateBuffer();
        EWSignals pull_ew();
        bool pull_shift();
        void connect_shift(StateBufferShiftInterface *);
        void step();
        StateBufferShiftInterface *north;
    private:
        RandomInterfaceGenerator rand_gen;
        ZeroInterfaceGenerator   zero_gen;
        //StateBufferShiftInterface *north;
        bool shift;
};

class StateBufferArray {
    public:
        StateBufferArray(int _num_buffers = 128);
        ~StateBufferArray();
        StateBuffer& operator[](int index);
        void step();
        int num();
        void connect_shift(StateBufferShiftInterface *);
    private:
        std::vector<StateBuffer> buffers;
        int num_buffers;
};


#endif 
