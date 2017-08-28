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
}

PSumBuffer::~PSumBuffer() {}

void
PSumBuffer::connect_west(EdgeInterface *_west) {
    west = _west;
}

void
PSumBuffer::set_address(addr_t addr) {
    ptrs.char_ptr = (char *)memory.translate(addr);

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
    int e_id = ew.psum_full_addr >> Constants::psum_buffer_width_bits;
    if (ew.column_countdown) {
        ARBPRECTYPE psum_dtype =  ew.psum_dtype; //get_upcast(ew.psum_dtype);
        if (ew.psum_start) {
            //assert(e_id < (int)entry.size());  FIXME - add range check
            //assert(entry[e_id].valid == false);  FIXME - add valid check
            switch (psum_dtype) {
                case R_UINT32:
                    ptrs.uint32_ptr[e_id] = 0;
                    break;
                case R_INT32:
                    ptrs.int32_ptr[e_id] = 0;
                    break;
                case R_UINT64:
                    ptrs.uint64_ptr[e_id] = 0;
                    break;
                case R_INT64:
                    ptrs.int64_ptr[e_id] = 0;
                    break;
                case R_FP32:
                    ptrs.fp32_ptr[e_id] = 0;
                    break;
                default:
                    assert(0);
            }
        }
        if (ew.ifmap_valid) {
            // assert(e_id < (int)entry.size());  FIXME - add range check
            // assert(entry[e_id].valid);  FIXME - add valid check
            printf("adding partial sum at %d is ", e_id);
            ArbPrec::dump(stdout, ns.partial_sum, psum_dtype);
            printf("\n");
            switch (psum_dtype) {
                case R_UINT32:
                    ptrs.uint32_ptr[e_id] = 
                        ArbPrec::add(&ptrs.uint32_ptr[e_id], &ns.partial_sum, psum_dtype).uint32;
                    break;
                case R_INT32:
                    ptrs.int32_ptr[e_id] = 
                        ArbPrec::add(&ptrs.int32_ptr[e_id], &ns.partial_sum, psum_dtype).int32;
                    break;
                case R_UINT64:
                    ptrs.uint64_ptr[e_id] = 
                        ArbPrec::add(&ptrs.uint64_ptr[e_id], &ns.partial_sum, psum_dtype).uint64;
                    break;
                case R_INT64:
                    ptrs.int64_ptr[e_id] = 
                        ArbPrec::add(&ptrs.int64_ptr[e_id], &ns.partial_sum, psum_dtype).int64;
                    break;
                case R_FP32:
                    ptrs.fp32_ptr[e_id] = 
                        ArbPrec::add(&ptrs.fp32_ptr[e_id], &ns.partial_sum, psum_dtype).fp32;
                    break;
                default:
                    assert(0);
            }
        }

        if (ew.psum_end) {
            //assert(entry[e_id].valid == true);   FIXME -add range check
            //assert(e_id < (int)entry.size()); FIXME - add range check
            printf("final partial sum at %d is ", e_id);
            ArbPrec::dump(stdout, ns.partial_sum, psum_dtype);
            printf("\n");
            //entry[e_id].valid = false; FIXME - set to invalid
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
