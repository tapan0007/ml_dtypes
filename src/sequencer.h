#ifndef SEQUENCER_H
#define SEQUENCER_H

#include "sigint.h"

class Sequencer : public SequencerInterface, public SbNSInterface  {
    public:
        Sequencer();
        ~Sequencer();
        void step();
        SbNSSignals pull_ns();
        bool pull_clamp();
        void set_clamp_time(tick_t delay);
        void set_opcode(Opcode op);
        void set_psum_addr(addr_t addr);
    private:
        addr_t   psum_addr;
        Opcode   op;
        tick_t clock;
        tick_t clamp_time;
};

#endif  
