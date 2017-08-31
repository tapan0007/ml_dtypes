#include "pool.h"
#include "io.h"

extern Memory memory;
extern addr_t psum_buffer_base;

//------------------------------------------------------------
// Pool
//------------------------------------------------------------
Pool::Pool() : connection(NULL) {
    ps = {0};
}

Pool::~Pool() {}

void
Pool::connect(PoolInterface *_connection) {
    connection = _connection;
}

PoolSignals
Pool::pull_pool() {
    return ps;
}

void
Pool::step() {
    if (!ps.pool_valid) {
        return;
    }
            //memory.write(ew.ofmap_full_addr, ArbPrec::element_ptr(ofmap_pixel, psum_dtype), (char *)ArbPrec::element_ptr(ofmap_pixel, psum_dtype) - (char *)&ofmap_pixel);
            //ew.ofmap_full_addr += Constants::partition_nbytes;
        //ew.column_countdown--;
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
}

#if 0
ArbPrecData
Pool::activation(ArbPrecData pixel) {
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
#endif



//------------------------------------------------------------
// PoolBufferArray
//------------------------------------------------------------
PoolArray::PoolArray(int n_cols) {
    pooler.resize(n_cols);
    for (int i = 1; i < n_cols; i++) {
        pooler[i].connect(&pooler[i-1]);
    }
}

PoolArray::~PoolArray() {}

void
PoolArray::connect(PoolInterface *connection) {
    pooler[0].connect(connection);
}


Pool& 
PoolArray::operator[](int index)
{
    return pooler[index];
}

void
PoolArray::step() {
    int n_cols = pooler.size();
    for (int i = n_cols - 1; i >= 0; i--) {
        pooler[i].step();
    }
}
