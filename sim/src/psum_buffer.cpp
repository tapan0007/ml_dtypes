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
 //   dtype = UINT32; // FIXME - should be arg?
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
    uint8_t valid;

    if (ew.column_valid && ew.ifmap_valid) {
        assert(ew.psum_addr.sys >= MMAP_PSUM_BASE);
        ARBPRECTYPE psum_dtype =  get_upcast(ew.ifmap_dtype);
        size_t dsize = sizeofArbPrecType(psum_dtype);
        unsigned int local_addr = ew.psum_addr.column;
        Addr src_addr;
        src_addr.sys = memory.index(mem_addr, local_addr, UINT8);
        Addr meta_addr;
        meta_addr.sys  = src_addr.sys | (1 << (COLUMN_BYTE_OFFSET_BITS + 
                    BANKS_PER_COLUMN_BITS));
        void *src_ptr = memory.translate(src_addr.sys);
        if (ew.psum_start) {
            memory.write(src_addr.sys, &zeros, dsize);
            memory.write(meta_addr.sys, &ones, dsize); 
        }
        if (ew.ifmap_valid) {
            printf("adding partial sum at %lx is ", mem_addr | src_addr.sys);
            ArbPrec::dump(stdout, ns.partial_sum, psum_dtype);
            ArbPrecData result = ArbPrec::add(src_ptr, &ns.partial_sum, psum_dtype);
            memory.write(src_addr.sys, &result, dsize);
            printf(" total = ");
            ArbPrec::dump(stdout, memory.translate(src_addr.sys), psum_dtype);
            printf("\n");
            memory.read(&valid, meta_addr.sys, 1);
            assert(valid);
        }

        if (ew.psum_stop) {
            printf("final partial sum at %lx is ", mem_addr | src_addr.sys);
            ArbPrec::dump(stdout, memory.translate(src_addr.sys), psum_dtype);
            printf("\n");
            memory.read(&valid, meta_addr.sys, 1);
            assert(valid);
            memory.write(meta_addr.sys, &zeros, 1); 
        } 

        if (ew.column_countdown == 0) {
            ew.column_valid = 0;
        } else {
            ew.column_countdown--;
        }
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
        col_buffer[i].set_address(psum_buffer_base + i * SZ(COLUMN_SIZE_BITS));
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
