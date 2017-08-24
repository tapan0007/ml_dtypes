#include "state_buffer.h"
#include "string.h"
#include "io.h"

extern Memory memory;
//------------------------------------------------------------------
// Single state buffer
//------------------------------------------------------------------
StateBuffer::StateBuffer() : north(nullptr), activate(nullptr), ns(), ifmap(nullptr), weights_rd(nullptr){
    memset(&ns, 0, sizeof(ns));
}

StateBuffer::~StateBuffer() {
}

ArbPrec
StateBuffer::read_addr(addr_t addr, ARBPRECTYPE type) {
    switch (type) {
        case UINT8:
            return ArbPrec(*((uint8_t *)memory.translate(addr)));
        case UINT32:
            return ArbPrec(*((uint32_t *)memory.translate(addr)));
        case FP32:
            return ArbPrec(*((float *)memory.translate(addr)));
        default:
            assert(0);
    }
    assert(0);
}

PeEWSignals 
StateBuffer::pull_ew() {
    ArbPrec weight = ArbPrec(uint8_t(0));
    ArbPrec pixel  = ArbPrec(uint8_t(0)); // send 0 pixels

    if (ns.ifmap_valid && ns.row_countdown) {
        //pixel = ArbPrec((uint8_t)(rand() % 0xff));
        pixel = read_addr(ns.ifmap_full_addr, ns.ifmap_dtype);
    }
    if (ns.weight_valid && ns.row_countdown) {
        //weight = ArbPrec((uint8_t)(rand() % 0xff));
        weight = read_addr(ns.weight_full_addr, ns.weight_dtype);
        printf("WEIGHT %d\n", weight.uint8);
    }

    return PeEWSignals{pixel, weight, ns.weight_dtype, ns.weight_toggle};
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
    if (e.row_countdown) {
        if (e.ifmap_valid) {
            e.ifmap_full_addr += Constants::partition_nbytes;
        } 
        if (e.weight_valid) {
            e.weight_full_addr += Constants::partition_nbytes;
        }
        e.row_countdown--;
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

void 
StateBuffer::step_write() {
    //ActivateSbSignals as = activate->pull_activate();
    //if (as.valid) {
    //    printf("sbact");
    //    as.partial_sum.dump(stdout);
     //   printf("\n");
    //}
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

void StateBufferArray::step_write() {
    for (int i = num() - 1; i >= 0; i--) {
        buffers[i].step_write();
    }
}
