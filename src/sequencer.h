#ifndef SEQUENCER_H
#define SEQUENCER_H

#include "sigint.h"
#include <queue>

typedef struct ConvolveArgs{
    addr_t ifmap_addr, filter_addr, ofmap_addr;
    ARBPRECTYPE weight_dtype;
    int i_n, i_c, i_h, i_w;
    int w_c, w_m, w_r, w_s;

} ConvolveArgs;

class Sequencer : public EdgeInterface  {
    public:
        Sequencer();
        ~Sequencer();
        void step();
        EdgeSignals pull_edge();
        void convolve(const ConvolveArgs &args);
        int steps_to_do();
        void dump();
    private:
        tick_t   clock;
        std::queue<EdgeSignals> uop;
        void dump_es(const EdgeSignals &es, bool header);

};

#endif  
