#include "state_buffer.h"

//------------------------------------------------------------------
// Single state buffer
//------------------------------------------------------------------
StateBuffer::StateBuffer() : north(NULL){
}

StateBuffer::~StateBuffer() {
}

PeEWSignals 
StateBuffer::pull_ew() {
    ArbPrec weight = ArbPrec((uint8_t)(0));
    ArbPrec pixel  = ArbPrec((uint8_t)(0));

    if (ns.ifmap_valid) {
        pixel = ArbPrec((uint8_t)(rand() % 0xff));
    }
    if (ns.weight_valid) {
        weight = ArbPrec((uint8_t)(rand() % 0xff));
    }

    return PeEWSignals(pixel, weight, ns.toggle_weight);
}

SbNSSignals 
StateBuffer::pull_ns() {
    return ns;
}

void 
StateBuffer::step() {
    ns = north->pull_ns();
}

void
StateBuffer::connect_north(SbNSInterface *_north) {
    north = _north;
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

int StateBufferArray::num() {
    return num_buffers;
}

void StateBufferArray::step() {
    for (int i = num_buffers - 1; i >= 0; i--) {
        buffers[i].step();
    }
}

StateBuffer& StateBufferArray::operator[](int index) {
    return buffers[index];
}

void
StateBufferArray::connect_north(SbNSInterface *north) {
    buffers[0].connect_north(north);
}
