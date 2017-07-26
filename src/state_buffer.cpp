#include "state_buffer.h"

//------------------------------------------------------------------
// Single state buffer
//------------------------------------------------------------------
StateBuffer::StateBuffer() : north(NULL), ifmap(NULL), weights(NULL){
}

StateBuffer::~StateBuffer() {
}

ArbPrec
StateBuffer::read_addr(void *addr, ArbPrecType type) {
    switch (type) {
        case UINT8:
            return ArbPrec(*((uint8_t *)addr));
        case UINT32:
            return ArbPrec(*((uint32_t *)addr));
        case FP32:
            return ArbPrec(*((float *)addr));
        default:
            assert(0);
    }
    assert(0);
}

void *
StateBuffer::inc_addr(void *addr, ArbPrecType type, int delta_index = 1) {
    char *ret_addr = static_cast<char *>(addr);
    switch (type) {
        case UINT8:
            ret_addr += sizeof(uint8_t) * delta_index;
            break;
        case UINT32:
            ret_addr += sizeof(uint32_t) * delta_index;
            break;
        case FP32:
            ret_addr += sizeof(float) * delta_index;
            break;
        default:
            assert(0);
    }
    return ret_addr;
}

PeEWSignals 
StateBuffer::pull_ew() {
    ArbPrec weight = ArbPrec((uint8_t)(0));
    ArbPrec pixel  = ArbPrec((uint8_t)(0));

    if (ns.ifmap_valid && ifmap != NULL) {
        //pixel = ArbPrec((uint8_t)(rand() % 0xff));
        pixel = read_addr(ifmap, UINT8);
        ifmap = (uint8_t *)inc_addr((void *)ifmap, UINT8);
    }
    if (ns.weight_valid && weights != NULL) {
        //weight = ArbPrec((uint8_t)(rand() % 0xff));
        weight = read_addr(weights, weights_type);
        printf("weight %d\n", ((uint8_t *)weights)[0]);
        weights = inc_addr(weights, weights_type);
    }

    return PeEWSignals(pixel, weight, ns.toggle_weight);
}

void
StateBuffer::connect_north(EdgeInterface *_north) {
    north = _north;
}

void
StateBuffer::load_ifmap(uint8_t *_ifmap) {
    ifmap = _ifmap;
    ifmap_offset = 0;
}

void
StateBuffer::load_weights(void *_weights, ArbPrecType type) {
    weights = _weights;
    weights_offset = 0;
    weights_type = type;
}

EdgeSignals 
StateBuffer::pull_edge() {
    return ns;
}

bool 
StateBuffer::pull_clamp() {
    return ns.clamp_weights;
}

void 
StateBuffer::step() {
    ns = north->pull_edge();
}


//------------------------------------------------------------------
// Single buffer array
//------------------------------------------------------------------

StateBufferArray::StateBufferArray(int _num_buffers) : num_buffers(_num_buffers) {
    /* must resize it right away so it doesn't get moved around on us */
    buffers.resize(num_buffers);
    buffers.push_back(StateBuffer());
    for (int i = 1; i < num_buffers; i++) {
        buffers.push_back(StateBuffer());
        buffers[i].connect_north(&buffers[i-1]);
    }
}

StateBufferArray::~StateBufferArray() {
}

StateBuffer& StateBufferArray::operator[](int index) {
    return buffers[index];
}

void
StateBufferArray::connect_north(EdgeInterface *north) {
    buffers[0].connect_north(north);
}

void
StateBufferArray::load_ifmap(uint8_t *_ifmap, int start_id, int end_id, int stride) {
    uint8_t *ifmap = _ifmap;
    for (int i = start_id; i < end_id; i++) {
        buffers[i].load_ifmap(ifmap);
        ifmap += stride;
    }
}

void
StateBufferArray::load_weights(void *_weights, int start_id, int end_id, int stride, ArbPrecType type) {
    void *weights = _weights;
    for (int i = start_id; i < end_id; i++) {
        buffers[i].load_weights(weights, type);
        weights = static_cast<char *>(weights) + stride;
    }
}


int StateBufferArray::num() {
    return num_buffers;
}

void StateBufferArray::step() {
    for (int i = num_buffers - 1; i >= 0; i--) {
        buffers[i].step();
    }
}

