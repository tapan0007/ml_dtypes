#include "sequencer.h"

Sequencer::Sequencer() : edge_signals(), clock(0) {
}

Sequencer::~Sequencer() {
}

void
Sequencer::step() {
    clock++;
}


EdgeSignals
Sequencer::pull_edge() {
    return edge_signals;

}
