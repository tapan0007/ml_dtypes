#ifndef STATE_BUFFER_H
#define STATE_BUFFER_H

#include "sigint.h"
#include "types.h"
#include <vector>

class StateBuffer : public PeEWInterface, public EdgeInterface, public SbEWBroadcastInterface {
    public:
        StateBuffer();
        ~StateBuffer();
        PeEWSignals   pull_ew();
        EdgeSignals pull_edge();
        bool        pull_clamp();
        void connect_north(EdgeInterface *);
        void connect_activate(ActivateSbInterface *);
        void step_read();
        void step_write();
    private:
        EdgeInterface           *north;
        ActivateSbInterface     *activate;
        EdgeSignals              ns;
        ARBPRECTYPE              type;
        uint8_t                 *ifmap;
        void                    *weights_rd;
        void                    *weights_wr;
        ARBPRECTYPE              weights_type;
        ArbPrec   read_addr(addr_t addr, ARBPRECTYPE type);

};

class StateBufferArray {
    public:
        StateBufferArray(int _num_buffers = 128);
        ~StateBufferArray();
        StateBuffer& operator[](int index);
        void step_read();
        void step_write();
        int num();
        void connect_activate(int id, ActivateSbInterface *);
        void connect_north(EdgeInterface *);
    private:
        std::vector<StateBuffer> buffers;
        int num_buffers;
};


#endif 
