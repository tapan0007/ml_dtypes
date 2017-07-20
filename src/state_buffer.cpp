#include "state_buffer.h"

//------------------------------------------------------------------
// Single state buffer
//------------------------------------------------------------------
StateBuffer::StateBuffer() : north(NULL){
}

StateBuffer::~StateBuffer() {
}

EWSignals 
StateBuffer::pull_ew() {
    EWSignals ew;
    if (shift) {
        ew = rand_gen.pull_ew();
    } else {
        ew = zero_gen.pull_ew();
    }
    return ew;
}

bool 
StateBuffer::pull_shift() {
    return shift;
}

void 
StateBuffer::step() {
    shift = north->pull_shift();
}

void
StateBuffer::connect_shift(StateBufferShiftInterface *shifter) {
    north = shifter;
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
        buffers[i].connect_shift(&buffers[i-1]);
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
StateBufferArray::connect_shift(StateBufferShiftInterface *shifter) {
    buffers[0].connect_shift(shifter);
}
