#ifndef SEQUENCER_H
#define SEQUENCER_H

#include "sigint.h"

class Sequencer : public EdgeInterface  {
    public:
        Sequencer();
        ~Sequencer();
        void step();
        EdgeSignals pull_edge();
        EdgeSignals edge_signals;
    private:
        tick_t   clock;
};

#endif  
