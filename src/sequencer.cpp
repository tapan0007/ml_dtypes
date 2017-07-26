#include "sequencer.h"

Sequencer::Sequencer() : psum_addr(0), clamp(false), clock(0) {
}

Sequencer::~Sequencer() {
}

void
Sequencer::step() {
    clock++;
    if (ifmap_valid) {
        psum_addr++;
    }
}

/* consider adding delay arg? */
void
Sequencer::set_psum_addr(addr_t _addr) {
    psum_addr = _addr;
}

void
Sequencer::set_ifmap_valid(bool truth) {
    ifmap_valid = truth;
}

void
Sequencer::set_toggle_weight(bool truth) {
   toggle_weight = truth;
}

void
Sequencer::set_weight_valid(bool truth) {
    weight_valid = truth;
}

void
Sequencer::set_clamp(bool truth) {
    clamp = truth;
}

EdgeSignals
Sequencer::pull_edge() {
    return EdgeSignals(ifmap_valid, weight_valid, toggle_weight, clamp, psum_addr);
}
