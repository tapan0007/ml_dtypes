#include "state_buffer.h"
#include "string.h"
#include "io.h"

//------------------------------------------------------------------
// Single state buffer
//------------------------------------------------------------------
StateBuffer::StateBuffer(MemoryMap *mmap, addr_t base, size_t nbytes) {
    this->ew.pixel_valid   = false;
    this->ew.weight_toggle = false;
    this->mem = mmap->mmap(base, nbytes);
}


PeEWSignals
StateBuffer::pull_ew() {
    return ew;
}

void
StateBuffer::connect_north(EdgeInterface *_north) {
    north = _north;
}

EdgeSignals
StateBuffer::pull_edge() {
    return ns;
}

bool
StateBuffer::pull_clamp() {
    return ns.weight_clamp;
}

void
StateBuffer::step_read() {
    ns = north->pull_edge();
    ArbPrecData weight;
    weight.raw = 0;
    ArbPrecData pixel;
    pixel.raw  = 0;
    bool pixel_valid = false;

    if (ns.row_valid) {
        if (ns.pad_valid) {
            pixel.raw = 0;
            pixel_valid = true;
        }
        if (ns.ifmap_valid) {
            mem->read_local_offset(&pixel.raw, ns.ifmap_addr.row,
                    sizeofArbPrecType(ns.ifmap_dtype));
            pixel_valid = true;
        }
        if (ns.weight_valid) {
            mem->read_local_offset(&weight.raw, ns.weight_addr.row,
                    sizeofArbPrecType(ns.weight_dtype));
            printf("WEIGHT %d\n", weight.uint8);
        }
    }

    ew = PeEWSignals{pixel, pixel_valid, weight, ns.weight_dtype, 
        ns.weight_toggle};

    if (ns.row_valid) {
        ns.row_valid = ((--ns.row_countdown) > 0);
    }
}

//------------------------------------------------------------------
// Single buffer array
//------------------------------------------------------------------

StateBufferArray::StateBufferArray(MemoryMap *mmap, addr_t base,
        int num_buffers) : corner_buffer(mmap, 0, 0) {
    /* FIXME, some of this range is reserved */
    size_t buffer_size = SZ(ROW_SIZE_BITS);
    for (int i = 0; i < num_buffers; i++) {
        buffers.push_back(StateBuffer(mmap, base, buffer_size));
        base += buffer_size;
    }

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

int StateBufferArray::num() {
    return buffers.size();
}

void StateBufferArray::step_read() {
    corner_buffer.step_read();
    for (int i = num() - 1; i >= 0; i--) {
        buffers[i].step_read();
    }
}

