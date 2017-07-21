#ifndef STATE_BUFFER_H
#define STATE_BUFFER_H

#include "sigint.h"
#include <vector>

class StateBuffer : public PeEWInterface, public SbNSInterface {
    public:
        StateBuffer();
        ~StateBuffer();
        PeEWSignals pull_ew();
        SbNSSignals pull_ns();
        void connect_north(SbNSInterface *);
        void step();
        SbNSInterface *north;
    private:
        SbNSSignals              ns;
};

class StateBufferArray {
    public:
        StateBufferArray(int _num_buffers = 128);
        ~StateBufferArray();
        StateBuffer& operator[](int index);
        void step();
        int num();
        void connect_north(SbNSInterface *);
    private:
        std::vector<StateBuffer> buffers;
        int num_buffers;
};


#endif 
