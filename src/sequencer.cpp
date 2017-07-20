#include "sequencer.h"

Sequencer::Sequencer() {
}

Sequencer::~Sequencer() {
}

void
Sequencer::step() {
    clock++;
}

/* consider adding delay arg? */
void
Sequencer::set_shift(bool truth) {
    shift = truth;
}

bool
Sequencer::pull_shift() {
    return shift;
}


