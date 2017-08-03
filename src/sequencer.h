#ifndef SEQUENCER_H
#define SEQUENCER_H

#include "sigint.h"
#include <deque>

typedef struct ConvolveArgs{
    addr_t ifmap_addr, filter_addr, ofmap_addr;
    ARBPRECTYPE weight_dtype;
    int i_r, i_s, i_t, i_u;
    int w_r, w_s, w_t, w_u;

} ConvolveArgs;

class Sequencer : public EdgeInterface  {
    public:
        Sequencer();
        ~Sequencer();
        void step();
        EdgeSignals pull_edge();
        void convolve(const ConvolveArgs &args);
        int steps_to_do();
    private:
        tick_t   clock;
        std::deque<EdgeSignals> uop;

};

#endif  
