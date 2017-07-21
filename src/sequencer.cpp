#include "sequencer.h"

Sequencer::Sequencer() : psum_addr(0), op(BUBBLE), clock(0), clamp_time(MAX_TICK) {
}

Sequencer::~Sequencer() {
}

void
Sequencer::step() {
    clock++;
    psum_addr++;
}

/* consider adding delay arg? */
void
Sequencer::set_psum_addr(addr_t _addr) {
    psum_addr = _addr;
}

void
Sequencer::set_opcode(Opcode _op) {
    op = _op;
}

void
Sequencer::set_clamp_time(tick_t delay) {
    clamp_time = clock + delay;
}

bool
Sequencer::pull_clamp() {
    return clamp_time == clock;
}

SbNSSignals
Sequencer::pull_ns() {
    return SbNSSignals(op, psum_addr);
}
