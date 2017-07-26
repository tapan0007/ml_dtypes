#ifndef SEQUENCER_H
#define SEQUENCER_H

#include "sigint.h"

class Sequencer : public EdgeInterface  {
    public:
        Sequencer();
        ~Sequencer();
        void step();
        EdgeSignals pull_edge();
        void set_clamp(bool truth);
        void set_ifmap_valid(bool truth);
        void set_weight_valid(bool truth);
        void set_toggle_weight(bool truth);
        void set_psum_addr(addr_t addr);
    private:
        addr_t   psum_addr;
        bool     ifmap_valid;
        bool     weight_valid;
        bool     toggle_weight;
        bool     clamp;
        tick_t   clock;
};

#endif  
