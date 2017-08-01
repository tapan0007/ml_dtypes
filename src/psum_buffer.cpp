#include "psum_buffer.h"
#include "string.h"
#include "io.h"

extern Memory memory;
//------------------------------------------------------------
// PsumBuffer
//------------------------------------------------------------
PSumBuffer::PSumBuffer(int n_entries) : ew(), north(nullptr), west(nullptr), ready_id(-1) {
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

ArbPrec
PSumBuffer::pool() {
    int e_id = ew.psum_id;
    ArbPrec pool_pixel = ArbPrec(ew.pool_dtype);
    int n = ew.pool_dimx * ew.pool_dimy;
    switch (ew.pool_type) {
        case AVG_POOL:
            // fixme - how can we divide with just a multiplying unit?
            //double pool_pixel = pool_pixel * ArbPrecType(ew.psum_dtype, (1.0 / (ew.pool_dimx * ew.pool_dimy)));
            for (int i = 0; i < ew.pool_dimx; i++) {
                for (int j = 0; j < ew.pool_dimy; j++) {
                    pool_pixel = pool_pixel + entry[e_id - i * ew.pool_stride - j].partial_sum;
                }
            }
            pool_pixel = pool_pixel / ArbPrec(ew.psum_dtype, n);
            break;
        case MAX_POOL:
            pool_pixel = entry[e_id].partial_sum;
            for (int i = 0; i < ew.pool_dimx; i++) {
                for (int j = 0; j < ew.pool_dimy; j++) {
                    ArbPrec comp_pixel = entry[e_id - i * ew.pool_stride - j].partial_sum;
                    if (comp_pixel > pool_pixel) {
                        pool_pixel = comp_pixel;
                    }
                }
            }
            break;
        case NO_POOL:
            pool_pixel = entry[e_id].partial_sum;
            break;
        default:
            break;
    }
    return pool_pixel;
}

ArbPrec
PSumBuffer::activation(ArbPrec pixel) {
    switch (ew.activation) {
        case RELU:
           break;
        case LEAKY_RELU:
           break;
        case SIGMIOD:
           break;
        case TANH:
           break;
        case IDENTITY:
           break;
        default:
           break;
    }
    return pixel;
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

        if (ew.pool_valid) {
            ArbPrec ofmap_pixel = pool();
            if (ew.activation_valid) {
                ofmap_pixel = activation(ofmap_pixel);
            }
            memory.write(ew.ofmap_addr, ofmap_pixel.raw_ptr(), ofmap_pixel.nbytes());
            ew.ofmap_addr += ew.ofmap_stride;
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
