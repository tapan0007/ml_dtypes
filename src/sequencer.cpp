#include "sequencer.h"

Sequencer::Sequencer() : row_countdown(0), column_countdown(0), psum_addr(0), psum_stride(0), psum_dtype(UINT8), start_psum(false), end_psum(0),  ifmap_valid(false), ifmap_addr(0), ifmap_stride(0), weight_valid(false), weight_addr(0), weight_stride(0), weight_dtype(UINT8), toggle_weight(false), clamp(false), clock(0) {
}

Sequencer::~Sequencer() {
}

void
Sequencer::step() {
    clock++;
    if (ifmap_valid) {
        psum_addr += sizeofArbPrecType(psum_dtype);
        ifmap_addr += sizeofArbPrecType(UINT8);
    }
    // subtract, because we are feeding backwards
    if (weight_valid) {
        weight_addr -= sizeofArbPrecType(weight_dtype);
    }
}


void
Sequencer::set_ifmap_addr(addr_t _addr) {
    ifmap_addr = _addr;
}

void
Sequencer::set_ifmap_stride(addr_t _stride) {
    ifmap_stride = _stride;
}

void
Sequencer::set_weight_addr(addr_t _addr) {
    weight_addr = _addr;
}

void
Sequencer::set_weight_stride(addr_t _stride) {
    weight_stride = _stride;
}

void
Sequencer::set_weight_dtype(ARBPRECTYPE _dtype) {
    weight_dtype = _dtype;
}

void
Sequencer::set_row_countdown(uint8_t _row_countdown) {
    row_countdown = _row_countdown;
}

void
Sequencer::set_column_countdown(uint8_t _column_countdown) {
    column_countdown = _column_countdown;
}

void
Sequencer::set_start_psum(bool truth) {
    start_psum = truth;
}

void
Sequencer::set_end_psum(bool truth) {
    end_psum = truth;
}

void
Sequencer::set_psum_addr(addr_t _addr) {
    psum_addr = _addr;
}

void
Sequencer::set_psum_stride(addr_t _stride) {
    psum_stride = _stride;
}

void
Sequencer::set_psum_dtype(ARBPRECTYPE _dtype) {
    psum_dtype = _dtype;
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
    EdgeSignals es = EdgeSignals{.row_countdown = row_countdown, .column_countdown = column_countdown, \
        .ifmap_valid = ifmap_valid, .ifmap_addr = ifmap_addr, .ifmap_stride = ifmap_stride, .weight_valid = weight_valid, .weight_addr = weight_addr, .weight_stride = weight_stride, weight_dtype = weight_dtype, .toggle_weight = toggle_weight, .clamp_weights = clamp,\
            .psum_addr = psum_addr, .psum_stride = psum_stride, .psum_dtype = psum_dtype, .start_psum = start_psum, .end_psum = end_psum};
    return es;

}
