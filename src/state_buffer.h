#ifndef STATE_BUFFER_H
#define STATE_BUFFER_H

#include "sigint.h"
#include "types.h"
#include <vector>

class StateBuffer : public PeEWInterface, public SbNSInterface, public SbEWBroadcastInterface {
    public:
        StateBuffer();
        ~StateBuffer();
        PeEWSignals pull_ew();
        SbNSSignals pull_ns();
        bool        pull_clamp();
        void connect_north(SbNSInterface *);
        void load_ifmap(uint8_t *ifmap);
        void load_weights(void *weights, ArbPrecType type);
        void step();
        SbNSInterface *north;
    private:
        SbNSSignals              ns;
        ArbPrecType              type;
        uint8_t                 *ifmap;
        int                      ifmap_offset;
        void                    *weights;
        ArbPrecType              weights_type;
        int                      weights_offset;
        ArbPrec   read_addr(void *addr, ArbPrecType type);
        void *    inc_addr(void *addr, ArbPrecType type, int index);

};

class StateBufferArray {
    public:
        StateBufferArray(int _num_buffers = 128);
        ~StateBufferArray();
        StateBuffer& operator[](int index);
        void step();
        int num();
        void connect_north(SbNSInterface *);
        void load_ifmap(uint8_t *ifmap, int start_id, int end_id, int stride);
        void load_weights(void *weights, int start_id, int end_id, int stride, ArbPrecType type);
    private:
        std::vector<StateBuffer> buffers;
        int num_buffers;
};


#endif 
