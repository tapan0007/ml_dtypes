#include "state_buffer.h"

StateBuffer::StateBuffer() {
}

StateBuffer::~StateBuffer() {
}

EWSignals StateBuffer::pull_ew() {
    return generator.pull_ew();
}

StateBufferArray::StateBufferArray(int _num_buffers) : num_buffers(_num_buffers) {
    for (int i = 0; i < num_buffers; i++) {
        buffers.push_back(StateBuffer());
    }
}

StateBufferArray::~StateBufferArray() {
}

int StateBufferArray::num() {
    return num_buffers;
}

StateBuffer& StateBufferArray::operator[](int index) {
    return buffers[index];
}
