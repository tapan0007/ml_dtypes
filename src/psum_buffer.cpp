#include "psum_buffer.h"
#include "string.h"
#include "io.h"

extern Memory memory;
extern addr_t psum_buffer_base;
//------------------------------------------------------------
// PsumBuffer
//------------------------------------------------------------
PSumBuffer::PSumBuffer() : ptr(NULL), ew(), north(nullptr), west(nullptr) {
    memset(&ns, 0, sizeof(ns));
    memset(&ew, 0, sizeof(ew));
    valids = (char *)(calloc(Constants::psum_addr, 1));
}

PSumBuffer::~PSumBuffer() {}

void
PSumBuffer::connect_west(EdgeInterface *_west) {
    west = _west;
}

void
PSumBuffer::set_address(addr_t addr) {
    mem_addr = addr;
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
 //   dtype = R_UINT32; // FIXME - should be arg?
 //   if (ready_id == -1) {
 //       return PSumActivateSignals{false, {0}, INVALID_ARBPRECTYPE};
 //   }
    //assert(0 && "this doesn't work");
    return PSumActivateSignals{false, {0}, INVALID_ARBPRECTYPE};
    //return PSumActivateSignals{ready_id != -1, ptrs.char_ptr[ready_id], dtype};
}

ArbPrecData
PSumBuffer::pool() {
//    int e_id = ew.psum_full_addr >> Constants::psum_buffer_width_bits;
 //   ArbPrecData pool_pixel = entry[e_id].partial_sum;
    // = ArbPrec(ew.pool_dtype);
    //int n = ew.pool_dimx * ew.pool_dimy;
#if 0
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
                    ArbPrecData comp_pixel = entry[e_id - i * Constants::partition_nbytes - j].partial_sum;
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
#endif
    return {0};
}

ArbPrecData
PSumBuffer::activation(ArbPrecData pixel) {
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
    static ArbPrecData zeros = {0};
    static ArbPrecData ones = {.uint64 = 0xffffffffffffffff};
    if (ew.column_countdown) {
        ARBPRECTYPE psum_dtype =  ew.psum_dtype; //get_upcast(ew.psum_dtype);
        size_t dsize = sizeofArbPrecType(psum_dtype);
        unsigned int e_id = ew.psum_full_addr >> Constants::psum_buffer_width_bits;
        unsigned int e_offset = e_id * dsize;
        addr_t src_addr = memory.index(mem_addr, e_id, psum_dtype);
        void *src_ptr = memory.translate(src_addr);
        assert(e_offset <= Constants::psum_addr);
        if (ew.psum_start) {
            memory.write(src_addr, &zeros, dsize);
            memcpy(&valids[e_offset], &ones, dsize); 
        }
        if (ew.ifmap_valid) {
            printf("adding partial sum at %d is ", e_id);
            ArbPrec::dump(stdout, ns.partial_sum, psum_dtype);
            printf("\n");
            ArbPrecData result = ArbPrec::add(src_ptr, &ns.partial_sum, psum_dtype);
            memory.write(src_addr, &result, dsize);
            assert(valids[e_offset]);
        }

        if (ew.psum_end) {
            printf("final partial sum at %d is ", e_id);
            ArbPrec::dump(stdout, ns.partial_sum, psum_dtype);
            printf("\n");
            assert(valids[e_offset]);
            memcpy(&valids[e_offset], &zeros, dsize); 
        } 

        if (ew.pool_valid) {
            ArbPrecData ofmap_pixel = pool();
            if (ew.activation_valid) {
                ofmap_pixel = activation(ofmap_pixel);
            }
            //memory.write(ew.ofmap_full_addr, ArbPrec::element_ptr(ofmap_pixel, psum_dtype), (char *)ArbPrec::element_ptr(ofmap_pixel, psum_dtype) - (char *)&ofmap_pixel);
            //ew.ofmap_full_addr += Constants::partition_nbytes;
        }
        ew.column_countdown--;
    }
}

//------------------------------------------------------------
// PsumBufferArray
//------------------------------------------------------------
PSumBufferArray::PSumBufferArray(int n_cols) {
    col_buffer.resize(n_cols);
    col_buffer[0].set_address(psum_buffer_base);
    for (int i = 1; i < n_cols; i++) {
        col_buffer[i].connect_west(&col_buffer[i-1]);
        col_buffer[i].set_address(psum_buffer_base + i * Constants::psum_addr);
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
