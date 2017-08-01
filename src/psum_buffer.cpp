#include "psum_buffer.h"
#include "string.h"

//------------------------------------------------------------
// PsumBuffer
//------------------------------------------------------------
PSumBuffer::PSumBuffer(int n_entries) : north(nullptr), west(nullptr), ready_id(-1) {
    PSumBufferEntry empty_entry = {.partial_sum = ArbPrec(uint32_t(0)), .valid=false};
    memset(&ns, 0, sizeof(ns));
    memset(&ew, 0, sizeof(ew));
    for (int i = 0; i < n_entries; i++) {
        entry.push_back(empty_entry);
    }
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

PSumActivateSignals
PSumBuffer::pull_psum() {
    return PSumActivateSignals{ready_id != -1, entry[ready_id].partial_sum};
}

void
PSumBuffer::step() {
    ns = north->pull_ns();
    ew = west->pull_edge();
    int e_id = ew.psum_id;
    assert(e_id < (int)entry.size());
    if (ew.column_countdown) {
        if (ew.psum_start) {
            entry[e_id].partial_sum = ArbPrec(ew.psum_dtype);
            entry[e_id].valid = true;

        }
        if (ew.ifmap_valid) {
            assert(entry[e_id].valid);
            entry[e_id].partial_sum = entry[e_id].partial_sum + ns.partial_sum;
        }

        if (ew.psum_end) {
            printf("partial sum at %d is ", e_id);
            entry[e_id].partial_sum.dump(stdout);
            entry[e_id].valid = false;
            printf("\n");
        } else {
            ready_id = -1;
        }
        ew.column_countdown--;
    }
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

PSumBuffer& 
PSumBufferArray::operator[](int index)
{
    return col_buffer[index];
}

void
PSumBufferArray::step() {
    int n_cols = col_buffer.size();
    for (int i = n_cols - 1; i >= 0; i--) {
        col_buffer[i].step();
    }
}
