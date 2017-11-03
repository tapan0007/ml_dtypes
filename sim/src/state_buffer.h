#ifndef STATE_BUFFER_H
#define STATE_BUFFER_H

#include "sigint.h"
#include "io.h"
#include <vector>

class StateBuffer : public PeEWInterface, public EdgeInterface, public SbEWBroadcastInterface {
    public:
        StateBuffer(MemoryMap *mmap, addr_t base, size_t nbytes);
        PeEWSignals   pull_ew();
        EdgeSignals pull_edge();
        bool        pull_clamp();
        void connect_north(EdgeInterface *);
        void step_read();
    private:
        EdgeInterface           *north        = nullptr;
        EdgeSignals              ns           = {0};
        MemoryMapInstance       *mem;

};

class StateBufferArray {
    public:
        StateBufferArray(MemoryMap *mmap, addr_t base, int num_buffers);
        ~StateBufferArray();
        StateBuffer& operator[](int index);
        void step_read();
        int num();
        void connect_north(EdgeInterface *);
        StateBuffer *get_edge();
    private:
        std::vector<StateBuffer> buffers;
        StateBuffer              corner_buffer;
};


#endif
