#include "psum_buffer.h"

//------------------------------------------------------------
// PsumBuffer
//------------------------------------------------------------
PSumBuffer::PSumBuffer(int n_entries) : north(nullptr), west(nullptr) {
    buffer.resize(n_entries);
}

PSumBuffer::~PSumBuffer() {}

void
PSumBuffer::connect_west(EdgeInterface *_west) {
    west = _west;
}

void
PSumBuffer::connect_north(PeNSInterface *_north) {
    north = _north;
}

EdgeSignals
PSumBuffer::pull_edge() {
    return ew;
}

void
PSumBuffer::step() {
    ns = north->pull_ns();
    ew = west->pull_edge();
}

//------------------------------------------------------------
// PsumBufferArray
//------------------------------------------------------------
PSumBufferArray::PSumBufferArray(int n_cols) {
    col_buffer.resize(n_cols);
    for (int i = 1; i < n_cols; i++) {
        col_buffer[i].connect_west(&col_buffer[i-1]);
    }
          
}

PSumBufferArray::~PSumBufferArray() {}

void
PSumBufferArray::connect_west(EdgeInterface *west) {
    col_buffer[0].connect_west(west);
}

void
PSumBufferArray::connect_north(int col, PeNSInterface *north) {
    col_buffer[col].connect_north(north);
}

void
PSumBufferArray::step() {
    int n_cols = col_buffer.size();
    for (int i = n_cols - 1; i >= 0; i--) {
        col_buffer[i].step();
    }
}
