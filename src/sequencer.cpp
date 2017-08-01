#include "sequencer.h"

Sequencer::Sequencer() : clock(0) {
}

Sequencer::~Sequencer() {
}

void
Sequencer::step() {
    clock++;
    if (edge_signals.ifmap_valid) {
        edge_signals.psum_id++;
        edge_signals.ifmap_addr += sizeofArbPrecType(UINT8);
    }
    // subtract, because we are feeding backwards
    if (edge_signals.weight_valid) {
        edge_signals.weight_addr -= sizeofArbPrecType(edge_signals.weight_dtype);
    }
}


EdgeSignals
Sequencer::pull_edge() {
    return edge_signals;

}
