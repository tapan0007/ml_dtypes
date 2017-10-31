#include "state_buffer.h"
#include "string.h"
#include "io.h"

extern Memory memory;
//------------------------------------------------------------------
// Single state buffer
//------------------------------------------------------------------
StateBuffer::StateBuffer() {
}


ArbPrecData
StateBuffer::read_addr(addr_t addr, ARBPRECTYPE type) {
    ArbPrecData ap;
    UNUSED(type);
    // assumes x86 little endianess
    ap.raw = *((uint64_t *)memory.translate(addr));
    return ap;
}

PeEWSignals 
StateBuffer::pull_ew() {
    ArbPrecData weight = {0};
    ArbPrecData pixel = {0};
    bool pixel_valid = false;

    if (ns.row_valid) {
        if (ns.pad_valid) {
            pixel = {0};
            pixel_valid = true;
        }
        if (ns.ifmap_valid) {
            pixel = read_addr(ns.ifmap_addr.sys, ns.ifmap_dtype);
            pixel_valid = true;
        }
        if (ns.weight_valid) {
            weight = read_addr(ns.weight_addr.sys, ns.weight_dtype);
            printf("WEIGHT %d\n", weight.uint8);
        }
    }

    return PeEWSignals{pixel, pixel_valid, weight, ns.weight_dtype, ns.weight_toggle};
}

void
StateBuffer::connect_north(EdgeInterface *_north) {
    north = _north;
}

void
StateBuffer::connect_activate(ActivateSbInterface *_activate) {
    activate = _activate;
}

EdgeSignals 
StateBuffer::pull_edge() {
    EdgeSignals e = ns;
    if (e.row_valid) {
        if (e.ifmap_valid) {
            e.ifmap_addr.sys += SZ(ROW_SIZE_BITS);
        } 
        if (e.weight_valid) {
            e.weight_addr.sys += SZ(ROW_SIZE_BITS);
        }
        if (e.row_countdown) {
            e.row_countdown--;
        } else {
            e.row_valid = false;
        }
    }
    return e;
}

bool 
StateBuffer::pull_clamp() {
    return ns.weight_clamp;
}

void 
StateBuffer::step_read() {
    ns = north->pull_edge();
}

//------------------------------------------------------------------
// Single buffer array
//------------------------------------------------------------------

StateBufferArray::StateBufferArray(int num_buffers) {
    /* must resize it right away so it doesn't get moved around on us */
    buffers.resize(num_buffers);
    // add a buffer for the corner
    for (int i = 1; i < num_buffers; i++) {
        buffers[i].connect_north(&buffers[i-1]);
    }
    corner_buffer.connect_north(&buffers[num_buffers-1]);
}

StateBufferArray::~StateBufferArray() {
}

StateBuffer& StateBufferArray::operator[](int index) {
    return buffers[index];
}

StateBuffer *
StateBufferArray::get_edge() {
    return &corner_buffer;
}

void
StateBufferArray::connect_north(EdgeInterface *north) {
    buffers[0].connect_north(north);
}

void
StateBufferArray::connect_activate(int id, ActivateSbInterface *activate) {
    buffers[id].connect_activate(activate);
}


int StateBufferArray::num() {
    return buffers.size();
}

void StateBufferArray::step_read() {
    corner_buffer.step_read();
    for (int i = num() - 1; i >= 0; i--) {
        buffers[i].step_read();
    }
}

